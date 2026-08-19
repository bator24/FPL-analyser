from __future__ import annotations

import numpy as np
import pandas as pd


def poisson_zero(lam: pd.Series | np.ndarray) -> np.ndarray:
    """P(0) under independent Poisson. Dixon-Coles rho is deferred (needs 0-0/1-0 calibration)."""
    lam = np.clip(pd.to_numeric(lam, errors="coerce").to_numpy(dtype=float), 0.05, 4.0)
    return np.exp(-lam)


def team_goals_against_lambda(frame: pd.DataFrame) -> pd.Series:
    """Blend our recent GA with opponent recent GF. Home sides concede a bit less."""
    index = frame.index

    def _col(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    our_ga = _col("opp_goals_r5")
    opp_gf = _col("opp_gf_r5")
    xgc = _col("expected_goals_conceded_r5")
    blended = (
        0.4 * our_ga.fillna(xgc).fillna(1.2)
        + 0.4 * opp_gf.fillna(our_ga).fillna(1.2)
        + 0.2 * xgc.fillna(our_ga).fillna(1.2)
    )
    home = frame.get("was_home")
    if home is not None:
        is_home = home.fillna(False).astype(bool)
        blended = blended.where(~is_home, blended * 0.9)
        blended = blended.where(is_home, blended * 1.1)
    return blended.clip(lower=0.2, upper=3.5)


def expected_clean_sheet(frame: pd.DataFrame, p_60: pd.Series) -> pd.Series:
    """FPL CS requires 60+ minutes. Team P(0 goals against) times P(60)."""
    p_team_cs = poisson_zero(team_goals_against_lambda(frame))
    return pd.Series(p_team_cs, index=frame.index) * pd.to_numeric(p_60, errors="coerce").fillna(0).clip(0, 1)
