"""Candidate pool for a single gameweek ILP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.config import Settings
from fpl.models.prior import (
    build_live_feature_frame,
    current_season_form_usable,
    ensure_code_map,
    next_unfinished_event,
    panel_has_gameweek,
    prior_season_key,
)
from fpl.models.xpts import structural_xpts
from fpl.optimize.rules import MAX_XPTS_IF_PLAYS, normalize_position


def with_risk_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Unconditional xPts plus an if-plays estimate for captaincy."""
    out = frame.copy()
    p_play = pd.to_numeric(out.get("p_play"), errors="coerce").fillna(0).clip(0, 1)
    xpts = pd.to_numeric(out.get("xpts"), errors="coerce").fillna(0).clip(lower=0)
    out["p_play"] = p_play
    out["p_zero"] = 1.0 - p_play
    out["xpts"] = xpts
    if "xpts_if_plays" in out.columns:
        if_plays = pd.to_numeric(out["xpts_if_plays"], errors="coerce")
        if_plays = if_plays.where(if_plays.notna(), xpts.where(p_play <= 1e-9, xpts / p_play.clip(lower=1e-9)))
    else:
        if_plays = xpts.where(p_play <= 1e-9, xpts / p_play.clip(lower=1e-9))
    out["xpts_if_plays"] = if_plays.clip(lower=0, upper=MAX_XPTS_IF_PLAYS)
    out["xpts_unconditional"] = xpts
    return out


def collapse_gameweek(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player: DGW xPts add, P(plays at least once) is a union."""
    if frame.empty:
        return frame
    work = frame.copy()
    work["element_id"] = pd.to_numeric(work["element_id"], errors="coerce")
    work = work.dropna(subset=["element_id"])
    work["element_id"] = work["element_id"].astype(int)
    work["position"] = work["position"].map(normalize_position)
    work = with_risk_columns(work)

    def _union(s: pd.Series) -> float:
        p = pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1)
        return float(1.0 - (1.0 - p).prod())

    agg: dict[str, Any] = {
        "xpts": "sum",
        "xpts_unconditional": "sum",
        "total_points": "sum",
        "p_play": _union,
        "cost_m": "max",
        "name": "first",
        "position": "first",
        "team": "first",
        "team_id": "first",
        "season": "first",
        "event": "first",
        "e_minutes": "sum",
        "news": "first",
        "status": "first",
        "chance_of_playing_next_round": "first",
        "form": "first",
        "transfers_in_event": "sum",
        "transfers_out_event": "sum",
        "event_points": "sum",
    }
    present = {k: v for k, v in agg.items() if k in work.columns}
    grouped = work.groupby("element_id", sort=False).agg(present).reset_index()
    grouped = with_risk_columns(grouped)
    return grouped


def default_season_event(panel: pd.DataFrame) -> tuple[str, int]:
    seasons = sorted(panel["season"].astype(str).unique())
    if not seasons:
        raise RuntimeError("player_gw has no seasons")
    season = seasons[-1]
    events = pd.to_numeric(panel.loc[panel["season"].astype(str) == season, "event"], errors="coerce")
    event = int(events.max())
    return season, event


def score_season(panel: pd.DataFrame, season: str) -> pd.DataFrame:
    season_rows = panel.loc[panel["season"].astype(str) == str(season)].copy()
    if season_rows.empty:
        raise RuntimeError(f"No player_gw rows for season {season}")
    if "xpts" not in season_rows.columns:
        season_rows = structural_xpts(season_rows)
    return season_rows


def score_gameweek(panel: pd.DataFrame, season: str, event: int) -> pd.DataFrame:
    season_rows = score_season(panel, season)
    gw = season_rows[pd.to_numeric(season_rows["event"], errors="coerce") == int(event)].copy()
    if gw.empty:
        raise RuntimeError(f"No player_gw rows for {season} GW{event}")
    return collapse_gameweek(gw)


def overlay_live_prices(pool: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Replace historical cost/club with the current bootstrap snapshot when ids match."""
    if players is None or players.empty:
        return pool
    cols = [c for c in ["element_id", "cost_m", "now_cost", "team_id", "web_name", "status"] if c in players.columns]
    live = players.loc[:, cols].copy()
    live["element_id"] = pd.to_numeric(live["element_id"], errors="coerce")
    live = live.dropna(subset=["element_id"])
    live["element_id"] = live["element_id"].astype(int)
    if "cost_m" not in live.columns and "now_cost" in live.columns:
        live["cost_m"] = pd.to_numeric(live["now_cost"], errors="coerce") / 10.0
    rename = {}
    if "web_name" in live.columns:
        rename["web_name"] = "name"
    live = live.rename(columns=rename)
    keep = ["element_id"] + [c for c in ["cost_m", "team_id", "name", "status"] if c in live.columns]
    live = live[keep].drop_duplicates("element_id")
    suffixes = ("", "_live")
    merged = pool.merge(live, on="element_id", how="left", suffixes=suffixes)
    for column in ("cost_m", "team_id", "name"):
        live_col = f"{column}_live" if f"{column}_live" in merged.columns else None
        if live_col and live_col in merged.columns:
            merged[column] = merged[live_col].where(merged[live_col].notna(), merged[column])
            merged = merged.drop(columns=[live_col])
    if "status" in merged.columns:
        merged = merged[merged["status"].fillna("a").astype(str).str.lower().ne("u")]
        merged = merged.drop(columns=["status"])
    return merged.reset_index(drop=True)


@dataclass
class PoolLoad:
    pool: pd.DataFrame
    season: str
    event: int
    source: str
    note: str
    n_mapped: int = 0
    n_unmapped: int = 0
    form_season: str | None = None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _live_tables(settings: Settings) -> dict[str, pd.DataFrame] | None:
    players = _read_parquet(settings.processed_dir / "players.parquet")
    fixtures = _read_parquet(settings.processed_dir / "fixtures.parquet")
    if players is None or fixtures is None or players.empty or fixtures.empty:
        return None
    teams = _read_parquet(settings.processed_dir / "teams.parquet")
    events = _read_parquet(settings.processed_dir / "events.parquet")
    return {
        "players": players,
        "fixtures": fixtures,
        "teams": teams if teams is not None else pd.DataFrame(),
        "events": events if events is not None else pd.DataFrame(),
    }


def _overrides(settings: Settings) -> pd.DataFrame:
    path = settings.overrides_dir / "xmins.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _live_note(*, season: str, event: int, form_season: str, map_by: str, n_mapped: int, n_unmapped: int) -> str:
    how = (
        f"{form_season} form mapped by FPL code"
        if map_by == "code"
        else f"{form_season} completed-match form (same ids)"
    )
    extra = ""
    if map_by == "code" and form_season != season:
        extra = (
            f" {season} player_gw is a stub, so form is {form_season} "
            "(not a handful of new IDs pretending to be a gameweek)."
        )
    return (
        f"Live pool for {season} GW{event}: {how}. "
        f"{n_mapped} players have PL history; {n_unmapped} are appearance-only "
        "(new to the league or unmapped). Prices, fixtures, and availability come from "
        "the FPL bootstrap (`news` / chance_of_playing), not a journalism scrape."
        f"{extra}"
    )


def load_prediction_pool(
    panel: pd.DataFrame,
    *,
    settings: Settings,
    season: str | None = None,
    event: int | None = None,
    overlay_live: bool = False,
) -> PoolLoad:
    """Score a GW for the ILP.

    Historical `--season/--event` that exist in `player_gw` use as-of rows (backtest).
    The live current season uses terminal form + next unfinished fixtures.
    """
    live = _live_tables(settings)
    if season is None:
        if live is not None:
            season = settings.current_season
        else:
            season, _ = default_season_event(panel)
    season = str(season)

    if event is None:
        if live is not None and season == settings.current_season:
            event = next_unfinished_event(live["fixtures"], live["events"])
        else:
            events = pd.to_numeric(
                panel.loc[panel["season"].astype(str) == season, "event"],
                errors="coerce",
            )
            if events.empty or events.isna().all():
                raise RuntimeError(
                    f"No player_gw rows for {season} and no live fixtures. "
                    "Run `python -m fpl ingest` and `python -m fpl history`."
                )
            event = int(events.max())
    event = int(event)

    use_history = panel_has_gameweek(panel, season, event)
    if (
        use_history
        and live is not None
        and season == settings.current_season
        and not current_season_form_usable(panel, season, n_live_players=int(len(live["players"])))
    ):
        use_history = False
    if use_history:
        pool = score_gameweek(panel, season, event)
        used_live_prices = False
        players_path = settings.processed_dir / "players.parquet"
        if overlay_live and players_path.exists() and season == settings.current_season:
            pool = overlay_live_prices(pool, pd.read_parquet(players_path))
            used_live_prices = True
        note = ""
        if used_live_prices:
            note = "Overlay: current bootstrap prices on a player_gw slice."
        return PoolLoad(pool=pool, season=season, event=event, source="player_gw", note=note)

    if live is None or season != settings.current_season:
        raise RuntimeError(f"No player_gw rows for {season} GW{event}")

    n_live = int(len(live["players"]))
    overlay_season: str | None = None
    if current_season_form_usable(panel, season, n_live_players=n_live):
        form_season = season
        map_by = "same_id"
        code_map = None
    else:
        form_season = prior_season_key(settings, panel, season)
        map_by = "code"
        code_map = ensure_code_map(settings, form_season)
        overlay_season = season

    features, stats = build_live_feature_frame(
        panel,
        live["players"],
        live["fixtures"],
        season=season,
        event=event,
        form_season=form_season,
        map_by=map_by,
        teams=live["teams"],
        code_map=code_map,
        overrides=_overrides(settings),
        overlay_season=overlay_season,
    )
    scored = structural_xpts(features)
    pool = collapse_gameweek(scored)
    note = _live_note(
        season=season,
        event=event,
        form_season=form_season,
        map_by=map_by,
        n_mapped=int(stats["n_mapped"]),
        n_unmapped=int(stats["n_unmapped"]),
    )
    return PoolLoad(
        pool=pool,
        season=season,
        event=event,
        source="live_prior",
        note=note,
        n_mapped=int(stats["n_mapped"]),
        n_unmapped=int(stats["n_unmapped"]),
        form_season=form_season,
    )
