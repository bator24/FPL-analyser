from pathlib import Path

import pandas as pd

from fpl.ingest.history import build_history_panel
from fpl.ingest.panel import normalize_merged_gw
from tests.helpers import make_settings


def test_normalize_maps_position_and_cost(tmp_path: Path) -> None:
    path = tmp_path / "merged_gw.csv"
    path.write_text(
        "name,position,team,element,fixture,opponent_team,was_home,kickoff_time,"
        "round,minutes,starts,total_points,goals_scored,assists,clean_sheets,"
        "goals_conceded,bonus,bps,value,xP,expected_goals,GW\n"
        "Haaland,FWD,Man City,123,10,2,True,2024-08-17T14:00:00Z,"
        "1,90,1,8,1,0,1,0,3,40,140,6.2,0.55,1\n",
        encoding="utf-8",
    )
    teams = tmp_path / "teams.csv"
    teams.write_text("id,name,short_name\n1,Man City,MCI\n2,Chelsea,CHE\n", encoding="utf-8")
    frame = normalize_merged_gw(path, "2024-25", teams)
    row = frame.iloc[0]
    assert row["position"] == "FWD"
    assert row["cost_m"] == 14.0
    assert row["played_60"] == 1
    assert row["fpl_xp_posthoc"] == 6.2
    assert row["team_id"] == 1


def test_history_panel_from_local_vaastav(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, history_seasons=("2024-25",))
    files = {
        "gws/merged_gw.csv": (
            "name,position,team,element,fixture,opponent_team,was_home,kickoff_time,"
            "round,minutes,starts,total_points,goals_scored,assists,clean_sheets,"
            "goals_conceded,own_goals,penalties_saved,penalties_missed,saves,"
            "bonus,bps,yellow_cards,red_cards,ict_index,influence,creativity,threat,"
            "expected_goals,expected_assists,expected_goal_involvements,"
            "expected_goals_conceded,value,selected,team_h_score,team_a_score,xP,GW\n"
            "Haaland,FWD,Man City,123,10,2,True,2024-08-17T14:00:00Z,"
            "1,90,1,8,1,0,1,0,0,0,0,0,3,40,0,0,12,8,2,9,0.55,0.1,0.65,0.2,140,1000,1,0,6.2,1\n"
            "Haaland,FWD,Man City,123,20,3,False,2024-08-24T14:00:00Z,"
            "2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,140,1000,2,0,4.0,2\n"
        ),
        "fixtures.csv": "id,team_h_difficulty,team_a_difficulty\n10,2,4\n20,3,3\n",
        "teams.csv": "id,name,short_name\n1,Man City,MCI\n2,Chelsea,CHE\n3,Arsenal,ARS\n",
    }

    def fake_fetch(url: str, *, timeout: int, user_agent: str) -> bytes:
        for rel, body in files.items():
            if url.endswith(rel) or url.endswith(Path(rel).name):
                return body.encode("utf-8")
        raise AssertionError(url)

    result = build_history_panel(
        settings=settings,
        refresh=True,
        include_current=False,
        fetch_bytes_fn=fake_fetch,
    )
    panel = result["player_gw"]
    assert len(panel) == 2
    gw2 = panel.sort_values("event").iloc[1]
    assert gw2["minutes_r1"] == 90
    assert gw2["fdr"] == 3
    assert (settings.processed_dir / "player_gw.parquet").exists()
