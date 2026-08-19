"""Map last completed PL form onto the live FPL squad and next fixtures.

As-of rolls in `player_gw` exclude the current row (correct for backtests). A
live next-GW prediction needs the *outcomes* of the last completed matches,
including the season finale, then this year's prices, fixtures, and FPL
availability (`news` / `chance_of_playing_*`). No journalism scrape.
"""

from __future__ import annotations

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


def apply_fpl_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Haircut minutes using official FPL status / chance_of_playing. Drop unavailable."""
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

    if "chance_of_playing_next_round" in out.columns:
        chance = pd.to_numeric(out["chance_of_playing_next_round"], errors="coerce")
    else:
        chance = pd.Series(np.nan, index=out.index, dtype=float)
    if "chance_of_playing_this_round" in out.columns:
        chance = chance.fillna(pd.to_numeric(out["chance_of_playing_this_round"], errors="coerce"))
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


def _fill_unmapped(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mapped = (
        out["mapped"].fillna(False).astype(bool)
        if "mapped" in out.columns
        else pd.Series(False, index=out.index)
    )
    if mapped.all():
        return out
    if "chance_of_playing_next_round" in out.columns:
        chance = pd.to_numeric(out["chance_of_playing_next_round"], errors="coerce")
    else:
        chance = pd.Series(np.nan, index=out.index, dtype=float)
    if "chance_of_playing_this_round" in out.columns:
        chance = chance.fillna(pd.to_numeric(out["chance_of_playing_this_round"], errors="coerce"))
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
