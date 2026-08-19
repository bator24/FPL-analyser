import pandas as pd

from fpl.features.rolling import add_player_rolling, add_team_opponent_rolling
from fpl.ingest.panel import add_match_goals


def _two_match_player() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "element_id": [1, 1],
            "event": [1, 2],
            "fixture_id": [10, 20],
            "team_id": [1, 1],
            "opponent_team": [2, 3],
            "kickoff_time": ["2024-08-17T14:00:00Z", "2024-08-24T14:00:00Z"],
            "minutes": [90, 0],
            "total_points": [8, 0],
            "starts": [1, 0],
            "goals_scored": [1, 0],
            "assists": [0, 0],
            "clean_sheets": [1, 0],
            "goals_conceded": [0, 2],
            "bonus": [2, 0],
            "bps": [30, 0],
            "expected_goals": [0.4, 0.0],
            "expected_assists": [0.1, 0.0],
            "expected_goals_conceded": [0.3, 1.8],
            "ict_index": [10.0, 0.0],
            "played": [1, 0],
            "played_60": [1, 0],
            "was_home": [True, False],
            "team_h_score": [1, 2],
            "team_a_score": [0, 0],
        }
    )


def test_rolling_features_do_not_leak_current_gw() -> None:
    featured = add_player_rolling(_two_match_player())
    gw1 = featured.iloc[0]
    gw2 = featured.iloc[1]
    assert pd.isna(gw1["minutes_r1"])
    assert gw2["minutes_r1"] == 90
    assert gw2["minutes_r5"] == 90
    assert gw2["total_points_r1"] == 8
    assert gw2["minutes_r1"] != 0


def test_team_form_is_shifted() -> None:
    panel = add_match_goals(_two_match_player())
    # Need a second team match so opponent join has a prior row.
    extra = panel.iloc[[0]].copy()
    extra["element_id"] = 99
    extra["team_id"] = 2
    extra["opponent_team"] = 1
    extra["minutes"] = 90
    panel = pd.concat([panel, extra], ignore_index=True)
    featured = add_team_opponent_rolling(add_player_rolling(panel))
    player_gw2 = featured[(featured["element_id"] == 1) & (featured["event"] == 2)].iloc[0]
    assert player_gw2["team_goals_r1"] == 1
