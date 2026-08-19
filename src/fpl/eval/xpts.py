from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl.eval.baselines import mae


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    err = actual[mask] - predicted[mask]
    return float(np.sqrt(np.mean(np.square(err))))


def spearman(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    corr, _ = spearmanr(actual[mask], predicted[mask])
    return float(corr)


def point_bucket(points: pd.Series) -> pd.Series:
    pts = pd.to_numeric(points, errors="coerce").fillna(0)
    buckets = pd.Series("haulers", index=pts.index)
    buckets = buckets.mask(pts == 0, "zeros")
    buckets = buckets.mask((pts > 0) & (pts <= 2), "blanks")
    buckets = buckets.mask((pts > 2) & (pts <= 4), "tickers")
    return buckets


def xpts_eval_table(
    actual_points: pd.Series,
    model_points: pd.Series,
    baselines: dict[str, pd.Series],
    positions: pd.Series | None = None,
) -> dict[str, Any]:
    actual = pd.to_numeric(actual_points, errors="coerce")
    model = pd.to_numeric(model_points, errors="coerce")
    buckets = point_bucket(actual)
    report: dict[str, Any] = {
        "n": int(len(actual)),
        "model": {
            "mae": mae(actual, model),
            "rmse": rmse(actual, model),
            "spearman": spearman(actual, model),
            "buckets": {},
        },
        "baselines": {},
    }
    for name, values in baselines.items():
        series = pd.to_numeric(values, errors="coerce")
        report["baselines"][name] = {
            "mae": mae(actual, series),
            "rmse": rmse(actual, series),
            "spearman": spearman(actual, series),
            "buckets": {},
        }
        for bucket in ("zeros", "blanks", "tickers", "haulers"):
            mask = buckets == bucket
            report["baselines"][name]["buckets"][bucket] = {
                "n": int(mask.sum()),
                "rmse": rmse(actual[mask], series[mask]),
            }
    for bucket in ("zeros", "blanks", "tickers", "haulers"):
        mask = buckets == bucket
        report["model"]["buckets"][bucket] = {
            "n": int(mask.sum()),
            "rmse": rmse(actual[mask], model[mask]),
        }
    if positions is not None:
        report["by_position"] = {}
        for pos in ("GKP", "DEF", "MID", "FWD"):
            mask = positions.astype("string").str.upper().replace({"GK": "GKP"}) == pos
            if int(mask.sum()) == 0:
                continue
            report["by_position"][pos] = {
                "n": int(mask.sum()),
                "mae": mae(actual[mask], model[mask]),
                "spearman": spearman(actual[mask], model[mask]),
            }
    r5 = report["baselines"].get("total_points_r5", {})
    report["beats_r5_mae"] = report["model"]["mae"] < r5.get("mae", float("inf"))
    report["beats_r5_spearman"] = report["model"]["spearman"] > r5.get("spearman", float("-inf"))
    report["kill_pass"] = bool(report["beats_r5_mae"] and report["beats_r5_spearman"])
    return report
