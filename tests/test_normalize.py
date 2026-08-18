from fpl.ingest.normalize import normalize


def _bootstrap() -> dict:
    return {
        "teams": [
            {
                "id": 1,
                "name": "Arsenal",
                "short_name": "ARS",
                "code": 3,
                "strength": 4,
                "strength_overall_home": 1350,
                "strength_overall_away": 1330,
                "strength_attack_home": 80,
                "strength_attack_away": 78,
                "strength_defence_home": 80,
                "strength_defence_away": 79,
                "pulse_id": 1,
            }
        ],
        "element_types": [
            {
                "id": 4,
                "singular_name": "Forward",
                "singular_name_short": "FWD",
                "squad_select": 3,
                "squad_min_play": 1,
                "squad_max_play": 3,
            }
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2026-08-15T17:30:00Z",
                "average_entry_score": 0,
                "finished": False,
                "data_checked": False,
                "is_previous": False,
                "is_current": True,
                "is_next": False,
                "highest_score": None,
            }
        ],
        "elements": [
            {
                "id": 123,
                "code": 999,
                "first_name": "Erling",
                "second_name": "Haaland",
                "web_name": "Haaland",
                "team": 1,
                "team_code": 3,
                "element_type": 4,
                "now_cost": 140,
                "selected_by_percent": "55.0",
                "status": "a",
                "news": "",
                "news_added": None,
                "chance_of_playing_this_round": 100,
                "chance_of_playing_next_round": 100,
                "ep_this": "8.2",
                "ep_next": "7.4",
                "form": "0.0",
                "points_per_game": "0.0",
                "total_points": 0,
                "minutes": 0,
                "starts": 0,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 0,
                "goals_conceded": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "saves": 0,
                "bonus": 0,
                "bps": 0,
                "influence": "0.0",
                "creativity": "0.0",
                "threat": "0.0",
                "ict_index": "0.0",
                "expected_goals": "0.00",
                "expected_assists": "0.00",
                "expected_goal_involvements": "0.00",
                "expected_goals_conceded": "0.00",
            }
        ],
    }


def test_normalize_players_cost_and_position() -> None:
    tables = normalize(
        _bootstrap(),
        [
            {
                "id": 10,
                "code": 1,
                "event": 1,
                "finished": False,
                "kickoff_time": "2026-08-16T14:00:00Z",
                "minutes": 0,
                "started": False,
                "team_a": 2,
                "team_h": 1,
                "team_a_score": None,
                "team_h_score": None,
                "team_h_difficulty": 3,
                "team_a_difficulty": 4,
                "pulse_id": 99,
            }
        ],
    )
    player = tables["players"].iloc[0]
    assert player["element_id"] == 123
    assert player["position"] == "FWD"
    assert player["cost_m"] == 14.0
    assert player["ep_next"] == 7.4
    assert len(tables["teams"]) == 1
    assert len(tables["fixtures"]) == 1
    assert tables["fixtures"].iloc[0]["fixture_id"] == 10
