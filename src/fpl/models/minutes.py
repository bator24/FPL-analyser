from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError

SAFE_FEATURE_COLUMNS = [
    "minutes_lag1",
    "minutes_r1",
    "minutes_r3",
    "minutes_r5",
    "minutes_r10",
    "minutes_r38",
    "starts_lag1",
    "starts_r3",
    "starts_r5",
    "played_lag1",
    "played_r3",
    "played_r5",
    "played_r10",
    "played_60_lag1",
    "played_60_r3",
    "played_60_r5",
    "was_home_int",
    "fdr",
    "cost_m",
    "event",
    "pos_gkp",
    "pos_def",
    "pos_mid",
    "pos_fwd",
]

DEFAULT_TEST_SEASONS = ("2023-24", "2024-25", "2025-26")


def prepare_minutes_frame(player_gw: pd.DataFrame) -> pd.DataFrame:
    """Add model columns without touching current-GW outcomes used as targets."""
    out = player_gw.copy()
    if "was_home" in out.columns:
        out["was_home_int"] = out["was_home"].fillna(False).astype(bool).astype(int)
    else:
        out["was_home_int"] = 0
    position = out["position"].astype("string").str.upper() if "position" in out.columns else pd.Series("", index=out.index)
    out["pos_gkp"] = position.isin(["GKP", "GK"]).astype(int)
    out["pos_def"] = (position == "DEF").astype(int)
    out["pos_mid"] = (position == "MID").astype(int)
    out["pos_fwd"] = (position == "FWD").astype(int)
    out["played"] = (pd.to_numeric(out["minutes"], errors="coerce").fillna(0) > 0).astype(int)
    out["played_60"] = (pd.to_numeric(out["minutes"], errors="coerce").fillna(0) >= 60).astype(int)
    history_cols = [c for c in SAFE_FEATURE_COLUMNS if c.startswith(("minutes_", "starts_", "played_"))]
    for column in history_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    if "fdr" in out.columns:
        out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce")
    if "cost_m" in out.columns:
        out["cost_m"] = pd.to_numeric(out["cost_m"], errors="coerce")
    return out


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in SAFE_FEATURE_COLUMNS if c in frame.columns]
    missing = [c for c in SAFE_FEATURE_COLUMNS if c not in frame.columns]
    features = frame.loc[:, columns].copy()
    for column in missing:
        features[column] = np.nan
    return features.loc[:, SAFE_FEATURE_COLUMNS]


def combine_minutes(*, p_play: np.ndarray, minutes_if_play: np.ndarray) -> np.ndarray:
    return np.clip(p_play, 0.0, 1.0) * np.clip(minutes_if_play, 0.0, 90.0)


def combine_p60(*, p_play: np.ndarray, p_60_if_play: np.ndarray) -> np.ndarray:
    return np.clip(p_play, 0.0, 1.0) * np.clip(p_60_if_play, 0.0, 1.0)


@dataclass
class MinutesModel:
    """Play/60' classifiers plus a sticky last-GW minutes prior.

    A GBM on minutes (and a residual GBM on last-GW) lost to last-GW MAE
    on 2023–26 walk-forward. We keep that finding: do not use a worse model.
    """

    play: HistGradientBoostingClassifier
    sixty: HistGradientBoostingClassifier
    minutes_if_play: HistGradientBoostingRegressor

    @classmethod
    def unfitted(cls) -> "MinutesModel":
        hgb_c = dict(max_depth=6, max_iter=80, learning_rate=0.08, random_state=42)
        return cls(
            play=HistGradientBoostingClassifier(**hgb_c),
            sixty=HistGradientBoostingClassifier(**hgb_c),
            minutes_if_play=HistGradientBoostingRegressor(
                max_depth=6, max_iter=80, learning_rate=0.08, random_state=42
            ),
        )

    def fit(self, frame: pd.DataFrame) -> "MinutesModel":
        features = feature_matrix(frame)
        played = frame["played"].astype(int)
        minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
        played_60 = frame["played_60"].astype(int)
        self.play.fit(features, played)
        self.sixty.fit(features, played_60)
        if int(played.sum()) >= 20:
            self.minutes_if_play.fit(features.loc[played == 1], minutes.loc[played == 1])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = feature_matrix(frame)
        sticky = sticky_minutes(frame).to_numpy()
        try:
            p_play = self.play.predict_proba(features)[:, 1]
        except NotFittedError:
            p_play = np.full(len(frame), 0.5)
        try:
            p_60 = self.sixty.predict_proba(features)[:, 1]
        except NotFittedError:
            p_60 = np.full(len(frame), 0.5)
        try:
            minutes_if_play = self.minutes_if_play.predict(features)
        except NotFittedError:
            minutes_if_play = np.full(len(frame), 60.0)
        out = frame.copy()
        out["p_play"] = p_play
        out["p_zero"] = 1.0 - p_play
        out["minutes_if_play"] = np.clip(minutes_if_play, 0.0, 90.0)
        out["e_minutes"] = np.clip(sticky, 0.0, 90.0)
        out["p_60"] = np.clip(p_60, 0.0, 1.0)
        out["confidence"] = confidence_flag(out)
        return out


def confidence_flag(frame: pd.DataFrame) -> pd.Series:
    p_play = pd.to_numeric(frame["p_play"], errors="coerce")
    played_r5 = pd.to_numeric(frame.get("played_r5"), errors="coerce")
    flags = pd.Series("medium", index=frame.index)
    flags = flags.mask(p_play.between(0.35, 0.65), "low")
    flags = flags.mask((played_r5.fillna(0) < 0.4) & (p_play < 0.8), "low")
    flags = flags.mask((p_play >= 0.85) & (played_r5.fillna(0) >= 0.75), "high")
    flags = flags.mask(p_play.isna(), "low")
    return flags


def sticky_minutes(frame: pd.DataFrame) -> pd.Series:
    """Last match, then last-3 mean, then 0. Strong naive minutes prior."""
    lag1 = pd.to_numeric(frame.get("minutes_lag1"), errors="coerce")
    r3 = pd.to_numeric(frame.get("minutes_r3"), errors="coerce")
    return lag1.fillna(r3).fillna(0)


def baseline_minutes(frame: pd.DataFrame) -> dict[str, pd.Series]:
    r3 = pd.to_numeric(frame.get("minutes_r3"), errors="coerce")
    lag1 = pd.to_numeric(frame.get("minutes_lag1"), errors="coerce")
    r3_filled = r3.fillna(lag1).fillna(0)
    return {
        "minutes_r3": r3_filled,
        "minutes_lag1": lag1.fillna(0),
        "sticky": sticky_minutes(frame),
        "starter_heuristic": pd.Series(np.where(r3_filled >= 60, 90.0, 0.0), index=frame.index),
    }


def walk_forward_predict(
    player_gw: pd.DataFrame,
    test_seasons: tuple[str, ...] = DEFAULT_TEST_SEASONS,
) -> pd.DataFrame:
    """Train on strictly earlier seasons; score each test season out-of-sample."""
    prepared = prepare_minutes_frame(player_gw)
    chunks: list[pd.DataFrame] = []
    available = set(prepared["season"].astype(str).unique())
    for season in test_seasons:
        if season not in available:
            continue
        train = prepared[prepared["season"].astype(str) < season]
        test = prepared[prepared["season"].astype(str) == season]
        if train.empty or test.empty:
            continue
        model = MinutesModel.unfitted().fit(train)
        predicted = model.predict(test)
        predicted["fold"] = season
        chunks.append(predicted)
    if not chunks:
        raise RuntimeError("Walk-forward produced no test rows. Need at least one later season in player_gw.")
    return pd.concat(chunks, ignore_index=True)


def apply_xmins_overrides(
    predicted: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    current_season: str,
) -> pd.DataFrame:
    """Overrides apply to the current season only (FPL element ids reset each year)."""
    out = predicted.copy()
    if "override" not in out.columns:
        out["override"] = False
    if overrides is None or overrides.empty:
        return out
    if not {"element_id", "event"}.issubset(overrides.columns):
        return out
    clean = overrides.dropna(subset=["element_id", "event"]).copy()
    if clean.empty:
        return out
    clean["element_id"] = pd.to_numeric(clean["element_id"], errors="coerce")
    clean["event"] = pd.to_numeric(clean["event"], errors="coerce")
    clean = clean.dropna(subset=["element_id", "event"])
    keep = [c for c in ["element_id", "event", "p_play", "e_minutes", "p_60", "source", "note"] if c in clean.columns]
    clean = clean[keep]
    current_mask = out["season"].astype(str) == str(current_season)
    if not current_mask.any():
        return out
    current = out.loc[current_mask].merge(clean, on=["element_id", "event"], how="left", suffixes=("", "_ov"))
    touched = pd.Series(False, index=current.index)
    for column in ("p_play", "e_minutes", "p_60"):
        ov_col = f"{column}_ov"
        if ov_col not in current.columns:
            continue
        use = current[ov_col].notna()
        current.loc[use, column] = pd.to_numeric(current.loc[use, ov_col], errors="coerce")
        touched = touched | use
    current.loc[touched, "override"] = True
    current["p_zero"] = 1.0 - pd.to_numeric(current["p_play"], errors="coerce")
    current = current.drop(columns=[c for c in current.columns if c.endswith("_ov")])
    current = current.reindex(columns=list(out.columns))
    rest = out.loc[~current_mask]
    return pd.concat([rest, current], ignore_index=True)
