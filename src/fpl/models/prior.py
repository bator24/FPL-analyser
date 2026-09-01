"""Map last completed PL form onto the live FPL squad and next fixtures.

As-of rolls in `player_gw` exclude the current row (correct for backtests). A
live next-GW prediction needs the *outcomes* of the last completed matches,
including the season finale, then this year's prices, fixtures, and FPL
availability (`news` / `chance_of_playing_*`). No journalism scrape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.config import Settings
from fpl.ingest.panel import POSITION_MAP
from fpl.ingest.vaastav import download_players_raw

FORM_MEAN_COLUMNS = (
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "played",
    "played_60",
    "saves",
    "yellow_cards",
)

UNAVAILABLE_STATUS = frozenset({"u", "n"})
UNMAPPED_DEFAULT_P_PLAY = 0.35
_NEWS_PCT = re.compile(r"(\d+)\s*%")
# After ~3 full appearances, this season's bootstrap rates fully replace last season.
THIS_SEASON_BLEND_MINUTES = 270.0
THIS_SEASON_BLEND_FLOOR = 45.0
THIS_SEASON_RATE_MAP = (
    ("expected_goals", "expected_goals_r5"),
    ("expected_assists", "expected_assists_r5"),
    ("goals_scored", "goals_scored_r5"),
    ("assists", "assists_r5"),
    ("expected_goals_conceded", "expected_goals_conceded_r5"),
    ("clean_sheets", "clean_sheets_r5"),
    ("bonus", "bonus_r5"),
)
# A handful of element-summary rows is not a season. Switching the live pool
# onto that stub made every unmapped premium look like a 0.35 appearance.
FORM_COVERAGE_MIN_PLAYERS = 200


def next_unfinished_event(
    fixtures: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> int:
    """Soonest gameweek that still has an unfinished fixture."""
    fx = fixtures.copy()
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    finished = fx["finished"].fillna(False).astype(bool) if "finished" in fx.columns else False
    unfinished = fx.loc[~finished & fx["event"].notna()]
    if not unfinished.empty:
        return int(unfinished["event"].min())
    if events is not None and not events.empty and "is_next" in events.columns:
        nxt = events.loc[events["is_next"].fillna(False).astype(bool)]
        if not nxt.empty:
            event_col = "event" if "event" in nxt.columns else "id"
            return int(pd.to_numeric(nxt[event_col], errors="coerce").iloc[0])
    raise RuntimeError("No upcoming fixtures in the snapshot. Run `python -m fpl ingest --refresh`.")


def current_season_form_usable(
    panel: pd.DataFrame,
    season: str,
    *,
    n_live_players: int | None = None,
) -> bool:
    """True only when this season's player_gw covers a real squad, not a stub.

    18 newly-scraped IDs after GW1 is not a GW1 panel. Using it as the form
    source marks Haaland/Raya as unmapped and benches them on noise.
    """
    if panel is None or panel.empty:
        return False
    rows = panel.loc[panel["season"].astype(str) == str(season)]
    if rows.empty or "element_id" not in rows.columns:
        return False
    n = int(pd.to_numeric(rows["element_id"], errors="coerce").nunique())
    floor = FORM_COVERAGE_MIN_PLAYERS
    if n_live_players is not None and n_live_players > 0:
        floor = max(100, min(FORM_COVERAGE_MIN_PLAYERS, int(0.4 * int(n_live_players))))
    return n >= floor


def panel_has_gameweek(panel: pd.DataFrame, season: str, event: int) -> bool:
    if panel is None or panel.empty:
        return False
    mask = (panel["season"].astype(str) == str(season)) & (
        pd.to_numeric(panel["event"], errors="coerce") == int(event)
    )
    return bool(mask.any())


def prior_season_key(settings: Settings, panel: pd.DataFrame, current_season: str) -> str:
    if not panel.empty:
        earlier = sorted(
            s for s in panel["season"].astype(str).unique() if s < str(current_season)
        )
        if earlier:
            return earlier[-1]
    seasons = tuple(settings.history_seasons)
    return seasons[-1] if seasons else "2025-26"


def load_code_map(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8", low_memory=False)
    id_col = "id" if "id" in raw.columns else "element"
    if id_col not in raw.columns or "code" not in raw.columns:
        raise RuntimeError(f"{path} needs `id` (or `element`) and `code` columns")
    out = pd.DataFrame(
        {
            "prior_element_id": pd.to_numeric(raw[id_col], errors="coerce"),
            "code": pd.to_numeric(raw["code"], errors="coerce"),
        }
    ).dropna()
    out["prior_element_id"] = out["prior_element_id"].astype(int)
    out["code"] = out["code"].astype(int)
    return out.drop_duplicates("code", keep="last")


def ensure_code_map(settings: Settings, prior_season: str, *, refresh: bool = False) -> pd.DataFrame:
    path = download_players_raw(prior_season, settings=settings, refresh=refresh)
    if path is None or not path.exists():
        raise RuntimeError(
            f"Missing {prior_season} players_raw.csv (FPL code map). "
            "Need network once, or place the vaastav file under data/raw/vaastav/"
            f"{prior_season}/players_raw.csv"
        )
    return load_code_map(path)


def terminal_form(panel: pd.DataFrame, season: str, last_n: int = 5) -> pd.DataFrame:
    """Mean of the last `last_n` *completed* fixtures, including the season finale.

    Unlike as-of `minutes_r5` on the GW38 row, this uses GW38's minutes.
    """
    rows = panel.loc[panel["season"].astype(str) == str(season)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["element_id"] = pd.to_numeric(rows["element_id"], errors="coerce")
    rows = rows.dropna(subset=["element_id"])
    rows["element_id"] = rows["element_id"].astype(int)
    rows["event"] = pd.to_numeric(rows["event"], errors="coerce")
    rows["kickoff_time"] = pd.to_datetime(
        rows["kickoff_time"] if "kickoff_time" in rows.columns else pd.NaT,
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if "played" not in rows.columns:
        rows["played"] = (pd.to_numeric(rows.get("minutes"), errors="coerce").fillna(0) > 0).astype(int)
    if "played_60" not in rows.columns:
        rows["played_60"] = (pd.to_numeric(rows.get("minutes"), errors="coerce").fillna(0) >= 60).astype(int)
    rows = rows.sort_values(
        ["element_id", "event", "kickoff_time", "fixture_id"],
        kind="mergesort",
        na_position="last",
    )
    parts: list[pd.Series] = []
    for element_id, group in rows.groupby("element_id", sort=False):
        history = group.tail(last_n)
        last = history.iloc[-1]
        record: dict[str, Any] = {
            "prior_element_id": int(element_id),
            "minutes_lag1": _num(last.get("minutes")),
            "played_lag1": _num(last.get("played")),
            "played_60_lag1": _num(last.get("played_60")),
            "starts_lag1": _num(last.get("starts")),
        }
        for window, suffix in ((1, "r1"), (3, "r3"), (5, "r5")):
            chunk = history.tail(window)
            for column in FORM_MEAN_COLUMNS:
                if column not in chunk.columns:
                    continue
                record[f"{column}_{suffix}"] = float(pd.to_numeric(chunk[column], errors="coerce").mean())
        parts.append(pd.Series(record))
    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(parts)


def _num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def attach_event_fixtures(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    event: int,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per player-fixture in `event`. Blank-GW players stay in the pool with no fixture."""
    base = players.copy()
    base["element_id"] = pd.to_numeric(base["element_id"], errors="coerce")
    base = base.dropna(subset=["element_id"])
    base["element_id"] = base["element_id"].astype(int)
    base["team_id"] = pd.to_numeric(base["team_id"], errors="coerce")

    fx = fixtures.copy()
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    fx = fx.loc[fx["event"] == int(event)].copy()
    for column in ("team_h", "team_a", "fixture_id", "team_h_difficulty", "team_a_difficulty"):
        if column in fx.columns:
            fx[column] = pd.to_numeric(fx[column], errors="coerce")

    team_names: dict[int, str] = {}
    if teams is not None and not teams.empty and "team_id" in teams.columns:
        name_col = "name" if "name" in teams.columns else "short_name"
        if name_col in teams.columns:
            team_names = {
                int(tid): str(name)
                for tid, name in zip(
                    pd.to_numeric(teams["team_id"], errors="coerce"),
                    teams[name_col],
                )
                if pd.notna(tid)
            }

    if fx.empty:
        out = base.copy()
        out["fixture_id"] = pd.NA
        out["was_home"] = pd.NA
        out["fdr"] = pd.NA
        out["kickoff_time"] = pd.NaT
        out["opponent_team"] = pd.NA
        out["has_fixture"] = False
        out["team"] = out["team_id"].map(team_names)
        out["season_event"] = int(event)
        return out

    player_cols = list(base.columns)
    home = base.merge(fx, left_on="team_id", right_on="team_h", how="inner", suffixes=("", "_fx"))
    home["was_home"] = True
    home["fdr"] = home.get("team_h_difficulty")
    home["opponent_team"] = home.get("team_a")

    away = base.merge(fx, left_on="team_id", right_on="team_a", how="inner", suffixes=("", "_fx"))
    away["was_home"] = False
    away["fdr"] = away.get("team_a_difficulty")
    away["opponent_team"] = away.get("team_h")

    matched = pd.concat([home, away], ignore_index=True)
    matched["has_fixture"] = True
    keep_extra = [c for c in ["fixture_id", "kickoff_time", "was_home", "fdr", "opponent_team", "has_fixture"] if c in matched.columns]
    matched = matched.loc[:, [c for c in player_cols if c in matched.columns] + keep_extra]

    used_ids = set(matched["element_id"].astype(int)) if not matched.empty else set()
    blank = base.loc[~base["element_id"].isin(used_ids)].copy()
    if not blank.empty:
        blank["fixture_id"] = pd.NA
        blank["was_home"] = pd.NA
        blank["fdr"] = pd.NA
        blank["kickoff_time"] = pd.NaT
        blank["opponent_team"] = pd.NA
        blank["has_fixture"] = False
        matched = pd.concat([matched, blank], ignore_index=True)

    matched["team"] = matched["team_id"].map(team_names)
    if "web_name" in matched.columns and "name" not in matched.columns:
        matched["name"] = matched["web_name"]
    elif "web_name" in matched.columns:
        matched["name"] = matched["name"].where(matched["name"].notna(), matched["web_name"])
    return matched.reset_index(drop=True)


def chance_from_news(news: Any) -> float:
    """Official FPL news often encodes the % when `chance_of_playing_*` is null."""
    match = _NEWS_PCT.search(str(news or ""))
    if not match:
        return float("nan")
    return float(match.group(1))


def _play_chance_series(frame: pd.DataFrame) -> pd.Series:
    if "chance_of_playing_next_round" in frame.columns:
        chance = pd.to_numeric(frame["chance_of_playing_next_round"], errors="coerce")
    else:
        chance = pd.Series(np.nan, index=frame.index, dtype=float)
    if "chance_of_playing_this_round" in frame.columns:
        chance = chance.fillna(pd.to_numeric(frame["chance_of_playing_this_round"], errors="coerce"))
    if "news" in frame.columns:
        chance = chance.fillna(frame["news"].map(chance_from_news))
    return chance


def apply_fpl_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Haircut minutes using official FPL status / chance_of_playing / news %. Drop unavailable."""
    out = frame.copy()
    if "status" in out.columns:
        status = out["status"].fillna("a").astype(str).str.lower()
    else:
        status = pd.Series("a", index=out.index)
    out = out.loc[~status.isin(UNAVAILABLE_STATUS)].copy()
    if "status" in out.columns:
        status = out["status"].fillna("a").astype(str).str.lower()
    else:
        status = pd.Series("a", index=out.index)

    chance = _play_chance_series(out)
    cap = (chance / 100.0).clip(0, 1)
    injured = status.isin(["i", "d"])
    cap = cap.where(~(injured & cap.isna()), 0.25)
    cap = cap.where(~status.eq("s"), 0.0)

    for column in ("played_r5", "played_r3", "played_lag1", "played_60_r5", "played_60_lag1"):
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        out[column] = values.where(cap.isna(), np.minimum(values.fillna(0), cap))

    zero = cap.eq(0)
    if zero.any():
        for column in ("minutes_lag1", "minutes_r3", "minutes_r5"):
            if column in out.columns:
                out.loc[zero, column] = 0.0
    no_fix = ~out.get("has_fixture", pd.Series(True, index=out.index)).fillna(False).astype(bool)
    if no_fix.any():
        for column in ("played_r5", "played_lag1", "played_60_r5", "minutes_lag1", "minutes_r3", "minutes_r5"):
            if column in out.columns:
                out.loc[no_fix, column] = 0.0
    return out.reset_index(drop=True)


def apply_xmins_to_features(
    frame: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    event: int,
) -> pd.DataFrame:
    """Write presser overrides into the roll columns the scorer reads."""
    out = frame.copy()
    if overrides is None or overrides.empty:
        return out
    if not {"element_id", "event"}.issubset(overrides.columns):
        return out
    clean = overrides.dropna(subset=["element_id", "event"]).copy()
    clean["element_id"] = pd.to_numeric(clean["element_id"], errors="coerce")
    clean["event"] = pd.to_numeric(clean["event"], errors="coerce")
    clean = clean.dropna(subset=["element_id", "event"])
    clean = clean.loc[clean["event"] == int(event)]
    if clean.empty:
        return out
    overlay = clean.drop_duplicates("element_id", keep="last")
    rename = {
        col: f"{col}_ov"
        for col in ("p_play", "e_minutes", "p_60")
        if col in overlay.columns
    }
    overlay = overlay.rename(columns=rename)
    keep = ["element_id"] + list(rename.values())
    merged = out.merge(overlay[keep], on="element_id", how="left")
    if "p_play_ov" in merged.columns:
        use = merged["p_play_ov"].notna()
        p = pd.to_numeric(merged["p_play_ov"], errors="coerce")
        for column in ("played_r5", "played_r3", "played_lag1"):
            if column in merged.columns:
                merged.loc[use, column] = p.loc[use]
    if "p_60_ov" in merged.columns:
        use = merged["p_60_ov"].notna()
        p60 = pd.to_numeric(merged["p_60_ov"], errors="coerce")
        if "played_60_r5" in merged.columns:
            merged.loc[use, "played_60_r5"] = p60.loc[use]
    if "e_minutes_ov" in merged.columns:
        use = merged["e_minutes_ov"].notna()
        mins = pd.to_numeric(merged["e_minutes_ov"], errors="coerce")
        for column in ("minutes_lag1", "minutes_r3", "minutes_r5"):
            if column in merged.columns:
                merged.loc[use, column] = mins.loc[use]
    merged["override"] = False
    if any(c in merged.columns for c in ("p_play_ov", "e_minutes_ov", "p_60_ov")):
        touched = False
        for column in ("p_play_ov", "e_minutes_ov", "p_60_ov"):
            if column in merged.columns:
                touched = touched | merged[column].notna()
        merged["override"] = touched
    return merged.drop(columns=[c for c in merged.columns if c.endswith("_ov")])


def overlay_form_by_element_id(live: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Prefer same-id current-season rolls where they exist (new signings with a GW row)."""
    if form is None or form.empty or live.empty:
        return live
    extra = form.rename(columns={"prior_element_id": "element_id"})
    if "element_id" not in extra.columns:
        return live
    extra["element_id"] = pd.to_numeric(extra["element_id"], errors="coerce")
    extra = extra.dropna(subset=["element_id"])
    extra["element_id"] = extra["element_id"].astype(int)
    extra = extra.drop_duplicates("element_id", keep="last")
    form_cols = [c for c in extra.columns if c != "element_id"]
    merged = live.merge(extra, on="element_id", how="left", suffixes=("", "_ov"))
    hit = pd.Series(False, index=merged.index)
    for col in form_cols:
        ov = f"{col}_ov"
        if ov not in merged.columns:
            continue
        has = merged[ov].notna()
        hit = hit | has
        if col not in merged.columns:
            merged[col] = merged[ov]
        else:
            merged[col] = merged[ov].where(has, merged[col])
        merged = merged.drop(columns=[ov])
    if "mapped" not in merged.columns:
        merged["mapped"] = False
    merged["mapped"] = merged["mapped"].fillna(False).astype(bool) | hit.fillna(False)
    return merged


def overlay_bootstrap_minutes(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Unmapped players who already played this season are not appearance-only 0.35."""
    out = frame.copy()
    if players is None or players.empty or "element_id" not in players.columns:
        return out
    boot = players.copy()
    boot["element_id"] = pd.to_numeric(boot["element_id"], errors="coerce")
    boot = boot.dropna(subset=["element_id"])
    boot["element_id"] = boot["element_id"].astype(int)
    mins = (
        pd.to_numeric(boot["minutes"], errors="coerce")
        if "minutes" in boot.columns
        else pd.Series(0.0, index=boot.index)
    )
    starts = (
        pd.to_numeric(boot["starts"], errors="coerce")
        if "starts" in boot.columns
        else pd.Series(0.0, index=boot.index)
    )
    boot["_boot_min"] = mins.fillna(0).clip(lower=0, upper=90)
    boot["_boot_start"] = starts.fillna(0)
    boot = boot.drop_duplicates("element_id", keep="last")
    merged = out.merge(boot[["element_id", "_boot_min", "_boot_start"]], on="element_id", how="left")
    mapped = (
        merged["mapped"].fillna(False).astype(bool)
        if "mapped" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    boot_min = merged["_boot_min"].fillna(0)
    use = (~mapped) & (boot_min > 0)
    started = use & (merged["_boot_start"].fillna(0) >= 1)
    p = (boot_min / 90.0).clip(0.35, 1.0)
    p = p.where(~started, 1.0)
    for column in ("played_r5", "played_r3", "played_lag1"):
        if column not in merged.columns:
            merged[column] = np.nan
        merged.loc[use, column] = p.loc[use]
    if "played_60_r5" not in merged.columns:
        merged["played_60_r5"] = np.nan
    merged.loc[use, "played_60_r5"] = np.where(
        boot_min.loc[use] >= 60, 1.0, (p.loc[use] * 0.7).to_numpy()
    )
    for column in ("minutes_lag1", "minutes_r3", "minutes_r5"):
        if column not in merged.columns:
            merged[column] = np.nan
        merged.loc[use, column] = boot_min.loc[use]
    if "mapped" not in merged.columns:
        merged["mapped"] = False
    merged.loc[use, "mapped"] = True
    lag1 = pd.to_numeric(merged.get("minutes_lag1"), errors="coerce").fillna(0)
    revive = mapped & (lag1 <= 0) & (boot_min >= 60)
    merged.loc[revive, "minutes_lag1"] = boot_min.loc[revive]
    if "played_lag1" in merged.columns:
        merged.loc[revive, "played_lag1"] = 1.0
    # GW1 is already in the bootstrap. Do not keep a last-season rotation haircut
    # on someone who just started 60+ this season (Haaland p_play 0.60 → benched C).
    started_now = (merged["_boot_start"].fillna(0) >= 1) & (boot_min >= 60)
    if bool(started_now.any()):
        for column in ("played_r5", "played_r3", "played_lag1", "played_60_r5"):
            if column not in merged.columns:
                merged[column] = np.nan
            cur = pd.to_numeric(merged[column], errors="coerce").fillna(0)
            merged.loc[started_now, column] = np.maximum(cur.loc[started_now].to_numpy(), 1.0)
        lag1 = pd.to_numeric(merged.get("minutes_lag1"), errors="coerce").fillna(0)
        merged.loc[started_now, "minutes_lag1"] = np.maximum(
            lag1.loc[started_now].to_numpy(), boot_min.loc[started_now].to_numpy()
        )
    return merged.drop(columns=["_boot_min", "_boot_start"])


def overlay_this_season_rates(frame: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blend this season's bootstrap xG/minutes into last-season rolls.

    Early 2026/27 would otherwise keep last year's sales forever, because the
    live prior is still 2025-26 form. `live` already carries bootstrap season
    totals (`minutes`, `expected_goals`, …) next to mapped `*_r5` columns.
    """
    out = frame.copy()
    if players is not None and not players.empty and "minutes" not in out.columns:
        boot = players.copy()
        boot["element_id"] = pd.to_numeric(boot["element_id"], errors="coerce")
        boot = boot.dropna(subset=["element_id"])
        boot["element_id"] = boot["element_id"].astype(int)
        cols = ["element_id", "minutes", "starts", *(src for src, _ in THIS_SEASON_RATE_MAP)]
        cols = [c for c in cols if c in boot.columns]
        boot = boot[cols].drop_duplicates("element_id", keep="last")
        out = out.merge(boot, on="element_id", how="left", suffixes=("", "_boot"))
        for src, _dest in THIS_SEASON_RATE_MAP:
            if src not in out.columns and f"{src}_boot" in out.columns:
                out[src] = out[f"{src}_boot"]
        if "minutes" not in out.columns and "minutes_boot" in out.columns:
            out["minutes"] = out["minutes_boot"]
        if "starts" not in out.columns and "starts_boot" in out.columns:
            out["starts"] = out["starts_boot"]
        out = out.drop(columns=[c for c in out.columns if c.endswith("_boot")])
    if "minutes" not in out.columns:
        return out
    boot_min = pd.to_numeric(out["minutes"], errors="coerce").fillna(0).clip(lower=0)
    starts = (
        pd.to_numeric(out["starts"], errors="coerce").fillna(0).clip(lower=0)
        if "starts" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    apps = starts.where(starts > 0, 1.0)
    weight = (boot_min / THIS_SEASON_BLEND_MINUTES).clip(0, 1)
    use = boot_min >= THIS_SEASON_BLEND_FLOOR
    if not bool(use.any()):
        return out
    play_now = pd.Series(np.where(boot_min >= 60, 1.0, (boot_min / 90.0).clip(0, 1)), index=out.index)
    p60_now = pd.Series(np.where(boot_min >= 60, 1.0, play_now * 0.7), index=out.index)
    w = weight
    for src, dest in THIS_SEASON_RATE_MAP:
        if src not in out.columns:
            continue
        if dest not in out.columns:
            out[dest] = np.nan
        per_app = pd.to_numeric(out[src], errors="coerce") / apps.replace(0, np.nan)
        old = pd.to_numeric(out[dest], errors="coerce")
        out.loc[use, dest] = ((1.0 - w) * old.fillna(per_app) + w * per_app).loc[use]
    for column, now in (
        ("played_r5", play_now),
        ("played_r3", play_now),
        ("played_lag1", play_now),
        ("played_60_r5", p60_now),
    ):
        if column not in out.columns:
            out[column] = np.nan
        old = pd.to_numeric(out[column], errors="coerce")
        out.loc[use, column] = ((1.0 - w) * old.fillna(now) + w * now).loc[use]
    return out


def _fill_unmapped(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mapped = (
        out["mapped"].fillna(False).astype(bool)
        if "mapped" in out.columns
        else pd.Series(False, index=out.index)
    )
    if mapped.all():
        return out
    chance = _play_chance_series(out)
    p = (chance / 100.0).clip(0, 1).fillna(UNMAPPED_DEFAULT_P_PLAY)
    fresh = ~mapped
    for column in ("played_r5", "played_r3", "played_lag1"):
        if column not in out.columns:
            out[column] = np.nan
        out.loc[fresh, column] = p.loc[fresh]
    if "played_60_r5" not in out.columns:
        out["played_60_r5"] = np.nan
    out.loc[fresh, "played_60_r5"] = (p.loc[fresh] * 0.7).clip(0, 1)
    minutes = (p * 60.0).clip(0, 90)
    for column in ("minutes_lag1", "minutes_r3", "minutes_r5"):
        if column not in out.columns:
            out[column] = np.nan
        out.loc[fresh, column] = minutes.loc[fresh]
    return out


def build_live_feature_frame(
    panel: pd.DataFrame,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    season: str,
    event: int,
    form_season: str,
    map_by: str,
    teams: pd.DataFrame | None = None,
    code_map: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
    overlay_season: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Live player rows with terminal form + this GW's fixtures. Not yet scored."""
    form = terminal_form(panel, form_season)
    live = players.copy()
    live["element_id"] = pd.to_numeric(live["element_id"], errors="coerce")
    live = live.dropna(subset=["element_id"])
    live["element_id"] = live["element_id"].astype(int)
    if "position" in live.columns:
        live["position"] = live["position"].astype("string").str.upper().map(POSITION_MAP).fillna(live["position"])

    if map_by == "code":
        if code_map is None or code_map.empty:
            raise RuntimeError("code_map is required when mapping last-season ids onto this year")
        live["code"] = pd.to_numeric(live.get("code"), errors="coerce")
        mapped_form = form.merge(code_map, on="prior_element_id", how="inner")
        form_cols = [c for c in mapped_form.columns if c not in {"prior_element_id"}]
        live = live.merge(mapped_form[form_cols], on="code", how="left")
        live["mapped"] = live["minutes_lag1"].notna() | live["played_r5"].notna()
    else:
        live = live.merge(
            form.rename(columns={"prior_element_id": "element_id"}),
            on="element_id",
            how="left",
        )
        live["mapped"] = live["minutes_lag1"].notna() | live["played_r5"].notna()

    if overlay_season and str(overlay_season) != str(form_season):
        live = overlay_form_by_element_id(live, terminal_form(panel, overlay_season))
    live = overlay_bootstrap_minutes(live, players)
    live = overlay_this_season_rates(live, players)
    n_mapped = int(live["mapped"].fillna(False).sum())
    n_unmapped = int((~live["mapped"].fillna(False)).sum())
    live = _fill_unmapped(live)
    live = attach_event_fixtures(live, fixtures, event, teams=teams)
    live["season"] = str(season)
    live["event"] = int(event)
    live = apply_fpl_availability(live)
    live = apply_xmins_to_features(live, overrides if overrides is not None else pd.DataFrame(), event=event)
    if "cost_m" not in live.columns and "now_cost" in live.columns:
        live["cost_m"] = pd.to_numeric(live["now_cost"], errors="coerce") / 10.0
    stats = {
        "n_mapped": n_mapped,
        "n_unmapped": n_unmapped,
        "form_season": form_season,
        "map_by": map_by,
        "n_rows": int(len(live)),
    }
    return live, stats
