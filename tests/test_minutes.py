import pandas as pd

from fpl.eval.baselines import minutes_eval_table
from fpl.models.minutes import (
    SAFE_FEATURE_COLUMNS,
    apply_xmins_overrides,
    combine_minutes,
    walk_forward_predict,
)


def test_features_exclude_current_minutes() -> None:
    assert "minutes" not in SAFE_FEATURE_COLUMNS
    assert "total_points" not in SAFE_FEATURE_COLUMNS
    assert "minutes_r3" in SAFE_FEATURE_COLUMNS


def test_mixture_haircuts_rotation_risk() -> None:
    # 50% chance of an 8.0-if-he-starts 90-minute haul is not 90 minutes.
    e_minutes = combine_minutes(p_play=[0.5], minutes_if_play=[90.0])
    assert e_minutes[0] == 45.0


def test_override_only_hits_current_season() -> None:
    predicted = pd.DataFrame(
        {
            "season": ["2025-26", "2026-27"],
            "element_id": [10, 10],
            "event": [1, 1],
            "p_play": [0.9, 0.9],
            "p_zero": [0.1, 0.1],
            "e_minutes": [80.0, 80.0],
            "p_60": [0.8, 0.8],
            "override": [False, False],
        }
    )
    overrides = pd.DataFrame(
        {
            "element_id": [10],
            "event": [1],
            "p_play": [0.2],
            "e_minutes": [15.0],
            "p_60": [0.1],
            "source": ["presser"],
            "note": ["rested"],
        }
    )
    out = apply_xmins_overrides(predicted, overrides, current_season="2026-27")
    current = out[out["season"] == "2026-27"].iloc[0]
    past = out[out["season"] == "2025-26"].iloc[0]
    assert current["e_minutes"] == 15.0
    assert current["p_play"] == 0.2
    assert bool(current["override"]) is True
    assert past["e_minutes"] == 80.0
    assert bool(past["override"]) is False


def test_walk_forward_trains_only_on_earlier_seasons() -> None:
    rows = []
    for season, minutes in [("2022-23", 90), ("2023-24", 0)]:
        for event in range(1, 25):
            rows.append(
                {
                    "season": season,
                    "event": event,
                    "element_id": 1,
                    "position": "MID",
                    "minutes": minutes,
                    "kickoff_time": f"{season[:4]}-08-01",
                    "was_home": True,
                    "fdr": 3,
                    "cost_m": 6.0,
                    "minutes_lag1": minutes,
                    "minutes_r1": minutes,
                    "minutes_r3": minutes,
                    "minutes_r5": minutes,
                    "minutes_r10": minutes,
                    "minutes_r38": minutes,
                    "starts_lag1": 1 if minutes else 0,
                    "starts_r3": 1 if minutes else 0,
                    "starts_r5": 1 if minutes else 0,
                    "played_lag1": 1 if minutes else 0,
                    "played_r3": 1 if minutes else 0,
                    "played_r5": 1 if minutes else 0,
                    "played_r10": 1 if minutes else 0,
                    "played_60_lag1": 1 if minutes else 0,
                    "played_60_r3": 1 if minutes else 0,
                    "played_60_r5": 1 if minutes else 0,
                }
            )
    panel = pd.DataFrame(rows)
    oos = walk_forward_predict(panel, test_seasons=("2023-24",))
    assert set(oos["season"].unique()) == {"2023-24"}
    assert "e_minutes" in oos.columns


def test_eval_always_includes_baselines() -> None:
    actual = pd.Series([0, 0, 90, 90])
    model = pd.Series([10.0, 5.0, 80.0, 85.0])
    p_play = pd.Series([0.2, 0.1, 0.9, 0.95])
    baselines = {"minutes_r3": pd.Series([40.0, 40.0, 90.0, 90.0])}
    report = minutes_eval_table(actual, model, p_play, baselines)
    assert "minutes_r3" in report["baselines"]
    assert report["beats_r3_mae_zeros"] is True
    assert "mae" in report["model"]
