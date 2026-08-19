from __future__ import annotations

import numpy as np
import pandas as pd


def scale_by_minutes(stat_r: pd.Series | None, minutes_r: pd.Series | None, e_minutes: pd.Series) -> pd.Series:
    """Scale a per-match rolling mean to expected minutes this GW."""
    expected = pd.to_numeric(e_minutes, errors="coerce").fillna(0)
    if stat_r is None:
        return pd.Series(0.0, index=expected.index)
    stat = pd.to_numeric(stat_r, errors="coerce").fillna(0)
    mins = pd.to_numeric(minutes_r, errors="coerce") if minutes_r is not None else pd.Series(np.nan, index=expected.index)
    denom = mins.where(mins >= 15, np.nan).fillna(90.0)
    return (stat * expected / denom).clip(lower=0)


def expected_bps(frame: pd.DataFrame, e_minutes: pd.Series) -> pd.Series:
    return scale_by_minutes(frame.get("bps_r5"), frame.get("minutes_r5"), e_minutes)


def bps_to_bonus(e_bps: pd.Series) -> pd.Series:
    """Match-relative bonus is noisy; this is a calibrated-enough map from expected BPS.

    Typical FPL: bonus mass sits above ~20 BPS. Linear ramp, cap at 3.
    """
    bps = pd.to_numeric(e_bps, errors="coerce").fillna(0)
    return ((bps - 18.0) / 12.0).clip(lower=0, upper=3)


def expected_bonus(frame: pd.DataFrame, e_minutes: pd.Series) -> pd.Series:
    """Prefer scaled historical bonus; fall back to BPS map; take a blend."""
    from_hist = scale_by_minutes(frame.get("bonus_r5"), frame.get("minutes_r5"), e_minutes).clip(0, 3)
    from_bps = bps_to_bonus(expected_bps(frame, e_minutes))
    return (0.6 * from_hist + 0.4 * from_bps).clip(0, 3)
