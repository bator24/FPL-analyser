from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_v = pd.to_numeric(actual, errors="coerce")
    predicted_v = pd.to_numeric(predicted, errors="coerce")
    mask = actual_v.notna() & predicted_v.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual_v[mask] - predicted_v[mask])))


def brier(actual: pd.Series | np.ndarray, probability: pd.Series | np.ndarray) -> float:
    actual_v = pd.to_numeric(actual, errors="coerce")
    probability_v = pd.to_numeric(probability, errors="coerce")
    mask = actual_v.notna() & probability_v.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean((probability_v[mask] - actual_v[mask]) ** 2))


def minutes_eval_table(
    actual_minutes: pd.Series,
    model_minutes: pd.Series,
    p_play: pd.Series,
    baselines: dict[str, pd.Series],
) -> dict[str, Any]:
    """Always compare the model to naive baselines. Zeros are first-class."""
    played = (actual_minutes.fillna(0) > 0).astype(float)
    zeros = actual_minutes.fillna(0) == 0
    report: dict[str, Any] = {
        "n": int(len(actual_minutes)),
        "n_zero": int(zeros.sum()),
        "zero_rate": float(zeros.mean()) if len(actual_minutes) else float("nan"),
        "model": {
            "mae": mae(actual_minutes, model_minutes),
            "mae_zeros": mae(actual_minutes[zeros], model_minutes[zeros]) if zeros.any() else float("nan"),
            "mae_played": mae(actual_minutes[~zeros], model_minutes[~zeros]) if (~zeros).any() else float("nan"),
            "brier_played": brier(played, p_play),
            "mean_p_play_on_zeros": float(p_play[zeros].mean()) if zeros.any() else float("nan"),
            "mean_p_play_on_played": float(p_play[~zeros].mean()) if (~zeros).any() else float("nan"),
        },
        "baselines": {},
    }
    for name, values in baselines.items():
        report["baselines"][name] = {
            "mae": mae(actual_minutes, values),
            "mae_zeros": mae(actual_minutes[zeros], values[zeros]) if zeros.any() else float("nan"),
            "mae_played": mae(actual_minutes[~zeros], values[~zeros]) if (~zeros).any() else float("nan"),
        }
    r3 = report["baselines"].get("minutes_r3", {})
    report["beats_r3_mae"] = report["model"]["mae"] < r3.get("mae", float("inf"))
    report["beats_r3_mae_zeros"] = report["model"]["mae_zeros"] < r3.get("mae_zeros", float("inf"))
    report["kill_pass"] = bool(report["beats_r3_mae_zeros"] and report["beats_r3_mae"])
    return report
