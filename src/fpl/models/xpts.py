from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.features.rolling import _shifted_rolling_mean
from fpl.models.bonus import expected_bonus, expected_bps, scale_by_minutes
from fpl.models.cs import expected_clean_sheet, team_goals_against_lambda
from fpl.models.minutes import sticky_minutes
from fpl.models.scoring import (
    ASSIST_POINTS,
    CS_POINTS,
    GC_BUNDLE,
    GOAL_POINTS,
    OWN_GOAL_POINTS,
    SAVE_BUNDLE,
    YELLOW_POINTS,
)

DEFAULT_TEST_SEASONS = ("2023-24", "2024-25", "2025-26")


def _pos(frame: pd.DataFrame) -> pd.Series:
    return frame["position"].astype("string").str.upper().replace({"GK": "GKP"})


def _goal_pts(position: pd.Series) -> pd.Series:
    return position.map(GOAL_POINTS).fillna(4).astype(float)


def _cs_pts(position: pd.Series) -> pd.Series:
    return position.map(CS_POINTS).fillna(0).astype(float)


def _ensure_extra_rolls(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["kickoff_time"] = pd.to_datetime(out.get("kickoff_time"), utc=True, errors="coerce")
    out = out.sort_values(["season", "element_id", "kickoff_time", "fixture_id"], kind="mergesort")
    grouped = out.groupby(["season", "element_id"], sort=False)
    for column in ("saves", "yellow_cards"):
        dest = f"{column}_r5"
        if dest in out.columns or column not in out.columns:
            continue
        lagged = grouped[column].shift(1)
        out[dest] = _shifted_rolling_mean(lagged, out["season"], out["element_id"], 5)
    return out


def minutes_priors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["e_minutes"] = sticky_minutes(out)
    played_r5 = pd.to_numeric(out.get("played_r5"), errors="coerce")
    played_lag = pd.to_numeric(out.get("played_lag1"), errors="coerce")
    p60 = pd.to_numeric(out.get("played_60_r5"), errors="coerce")
    out["p_play"] = played_r5.fillna(played_lag).fillna(0).clip(0, 1)
    out["p_60"] = p60.fillna(0).clip(0, 1)
    out["p_zero"] = 1.0 - out["p_play"]
    return out


def structural_xpts(frame: pd.DataFrame) -> pd.DataFrame:
    """Rule-based expected points. Uses as-of rolls only; current-GW outcomes are targets."""
    out = minutes_priors(_ensure_extra_rolls(frame))
    pos = _pos(out)
    e_min = out["e_minutes"]
    p_play = out["p_play"]
    p_60 = out["p_60"]

    xg = scale_by_minutes(out.get("expected_goals_r5"), out.get("minutes_r5"), e_min)
    xa = scale_by_minutes(out.get("expected_assists_r5"), out.get("minutes_r5"), e_min)
    # Pre-xG seasons: fall back to realised rolling goals/assists.
    g_fb = scale_by_minutes(out.get("goals_scored_r5"), out.get("minutes_r5"), e_min)
    a_fb = scale_by_minutes(out.get("assists_r5"), out.get("minutes_r5"), e_min)
    e_goals = xg.where(xg > 0, g_fb)
    e_assists = xa.where(xa > 0, a_fb)

    appear = p_play * 1.0 + p_60 * 1.0
    attack = e_goals * _goal_pts(pos) + e_assists * ASSIST_POINTS
    p_cs = expected_clean_sheet(out, p_60)
    defence_cs = p_cs * _cs_pts(pos)

    lam_ga = team_goals_against_lambda(out)
    gc_hit = pd.Series(
        np.where(pos.isin(["GKP", "DEF"]), p_60 * (lam_ga / GC_BUNDLE) * -1.0, 0.0),
        index=out.index,
    )
    save_pts = pd.Series(0.0, index=out.index)
    if "saves_r5" in out.columns:
        save_pts = scale_by_minutes(out["saves_r5"], out.get("minutes_r5"), e_min) / SAVE_BUNDLE
        save_pts = save_pts.where(pos.eq("GKP"), 0.0)
    yellow_pts = pd.Series(0.0, index=out.index)
    if "yellow_cards_r5" in out.columns:
        yellow_pts = scale_by_minutes(out["yellow_cards_r5"], out.get("minutes_r5"), e_min) * YELLOW_POINTS

    e_bonus = expected_bonus(out, e_min)
    e_bps = expected_bps(out, e_min)

    xpts = appear + attack + defence_cs + gc_hit + save_pts + yellow_pts + e_bonus
    # Haircut by p_zero already sits in p_play / p_60 / e_minutes. Clip nonsense.
    out["e_goals"] = e_goals
    out["e_assists"] = e_assists
    out["p_cs"] = p_cs
    out["e_bonus"] = e_bonus
    out["e_bps"] = e_bps
    out["xpts_structural"] = pd.Series(xpts, index=out.index).clip(lower=0, upper=20)
    out["xpts_r5"] = pd.to_numeric(out.get("total_points_r5"), errors="coerce").fillna(0)
    out["xpts"] = out["xpts_structural"]
    return out


def walk_forward_xpts(
    player_gw: pd.DataFrame,
    test_seasons: tuple[str, ...] = DEFAULT_TEST_SEASONS,
) -> pd.DataFrame:
    """Formula has no fitted weights; still score only later seasons so reports match minutes folds."""
    prepared = structural_xpts(player_gw)
    available = set(prepared["season"].astype(str).unique())
    parts = [prepared[prepared["season"].astype(str) == s] for s in test_seasons if s in available]
    if not parts:
        raise RuntimeError("No xPts test rows. Need player_gw seasons matching the walk-forward folds.")
    out = pd.concat(parts, ignore_index=True)
    out["fold"] = out["season"].astype(str)
    return out
