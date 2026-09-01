import numpy as np
import pandas as pd

from fpl.eval.xpts import point_bucket, xpts_eval_table
from fpl.models.bonus import bps_to_bonus, expected_bonus
from fpl.models.cs import expected_clean_sheet, poisson_zero
from fpl.models.xpts import structural_xpts


def test_poisson_zero_decreases_with_lambda() -> None:
    assert poisson_zero(pd.Series([0.1]))[0] > poisson_zero(pd.Series([1.5]))[0]


def test_cs_requires_sixty_minutes() -> None:
    frame = pd.DataFrame(
        {
            "opp_goals_r5": [1.0],
            "opp_gf_r5": [1.0],
            "expected_goals_conceded_r5": [1.0],
            "was_home": [True],
        }
    )
    assert expected_clean_sheet(frame, pd.Series([0.0])).iloc[0] == 0
    assert expected_clean_sheet(frame, pd.Series([1.0])).iloc[0] > 0


def test_bonus_is_capped() -> None:
    assert bps_to_bonus(pd.Series([80.0])).iloc[0] == 3
    frame = pd.DataFrame({"bonus_r5": [2.0], "minutes_r5": [90.0], "bps_r5": [40.0]})
    assert expected_bonus(frame, pd.Series([90.0])).iloc[0] <= 3


def test_structural_xpts_does_not_use_current_goals() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2024-25"],
            "element_id": [1],
            "event": [10],
            "position": ["FWD"],
            "minutes": [90],
            "total_points": [12],
            "goals_scored": [3],
            "assists": [0],
            "minutes_lag1": [90],
            "minutes_r5": [90],
            "played_r5": [1],
            "played_60_r5": [1],
            "expected_goals_r5": [0.4],
            "expected_assists_r5": [0.1],
            "goals_scored_r5": [0.4],
            "assists_r5": [0.1],
            "bonus_r5": [0.5],
            "bps_r5": [22],
            "total_points_r5": [4.0],
            "opp_goals_r5": [1.2],
            "opp_gf_r5": [1.1],
            "expected_goals_conceded_r5": [1.0],
            "was_home": [True],
            "kickoff_time": ["2024-10-01T12:00:00Z"],
            "fixture_id": [1],
        }
    )
    out = structural_xpts(frame)
    # Current-row hat-trick must not appear as ~12 xPts from goals_scored=3.
    assert out["xpts_structural"].iloc[0] < 10
    assert out["e_goals"].iloc[0] < 1.0


def test_structural_xpts_keeps_zero_xg() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2024-25"],
            "element_id": [1],
            "event": [10],
            "position": ["FWD"],
            "minutes_lag1": [90],
            "minutes_r5": [90],
            "played_r5": [1],
            "played_60_r5": [1],
            "expected_goals_r5": [0.0],
            "expected_assists_r5": [0.0],
            "goals_scored_r5": [1.5],
            "assists_r5": [0.8],
            "bonus_r5": [0.0],
            "bps_r5": [10],
            "total_points_r5": [2.0],
            "opp_goals_r5": [1.2],
            "opp_gf_r5": [1.1],
            "expected_goals_conceded_r5": [1.0],
            "was_home": [True],
            "kickoff_time": ["2024-10-01T12:00:00Z"],
            "fixture_id": [1],
        }
    )
    out = structural_xpts(frame)
    assert float(out["e_goals"].iloc[0]) == 0.0
    assert float(out["e_assists"].iloc[0]) == 0.0


def test_xpts_eval_includes_baselines_and_buckets() -> None:
    actual = pd.Series([0, 1, 3, 8])
    model = pd.Series([0.5, 1.5, 3.2, 6.0])
    report = xpts_eval_table(
        actual,
        model,
        {"total_points_r5": pd.Series([1.0, 1.0, 4.0, 5.0])},
        positions=pd.Series(["FWD", "MID", "MID", "FWD"]),
    )
    assert "total_points_r5" in report["baselines"]
    assert set(report["model"]["buckets"]) == {"zeros", "blanks", "tickers", "haulers"}
    assert point_bucket(actual).tolist() == ["zeros", "blanks", "tickers", "haulers"]
    assert "kill_pass" in report
    assert not np.isnan(report["model"]["spearman"])
