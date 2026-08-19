"""Candidate pool for a single gameweek ILP."""

from __future__ import annotations

from typing import Any

import pandas as pd

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
