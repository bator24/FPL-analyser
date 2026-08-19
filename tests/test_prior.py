import pandas as pd

from fpl.models.prior import (
    apply_fpl_availability,
    apply_xmins_to_features,
    attach_event_fixtures,
    build_live_feature_frame,
    next_unfinished_event,
    panel_has_gameweek,
    terminal_form,
)
from fpl.optimize.pool import load_prediction_pool
from tests.helpers import make_settings


def _gw(element_id: int, event: int, minutes: float, **extra) -> dict:
    row = {
        "season": "2025-26",
        "element_id": element_id,
        "event": event,
        "minutes": minutes,
        "played": int(minutes > 0),
        "played_60": int(minutes >= 60),
        "starts": int(minutes >= 60),
        "total_points": 2 if minutes else 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "bonus": 0,
        "bps": 10,
        "expected_goals": 0.2 if minutes else 0.0,
        "expected_assists": 0.1 if minutes else 0.0,
        "expected_goals_conceded": 1.0,
        "saves": 0,
        "yellow_cards": 0,
        "kickoff_time": f"2026-05-{event:02d}T12:00:00Z",
        "fixture_id": element_id * 100 + event,
        "name": f"P{element_id}",
        "position": "MID",
        "team_id": 1,
    }
    row.update(extra)
    return row


def test_terminal_form_includes_finale_minutes() -> None:
    """As-of rolls on GW38 exclude GW38; the live prior must not."""
    panel = pd.DataFrame(
        [
            _gw(10, 36, 0),
            _gw(10, 37, 0),
            _gw(10, 38, 90),
        ]
    )
    form = terminal_form(panel, "2025-26")
    assert form["minutes_lag1"].iloc[0] == 90
    assert form["played_lag1"].iloc[0] == 1
    assert form["minutes_r5"].iloc[0] == 30


def test_code_map_attaches_new_element_id() -> None:
    panel = pd.DataFrame([_gw(10, 38, 90)])
    players = pd.DataFrame(
        [
            {
                "element_id": 99,
                "code": 1000,
                "web_name": "Salah",
                "team_id": 1,
                "position": "MID",
                "now_cost": 145,
                "cost_m": 14.5,
                "status": "a",
                "chance_of_playing_next_round": 100,
            }
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "event": 1,
                "finished": False,
                "kickoff_time": "2026-08-15T14:00:00Z",
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            }
        ]
    )
    code_map = pd.DataFrame({"prior_element_id": [10], "code": [1000]})
    frame, stats = build_live_feature_frame(
        panel,
        players,
        fixtures,
        season="2026-27",
        event=1,
        form_season="2025-26",
        map_by="code",
        code_map=code_map,
    )
    assert stats["n_mapped"] == 1
    assert int(frame["element_id"].iloc[0]) == 99
    assert frame["minutes_lag1"].iloc[0] == 90
    assert bool(frame["was_home"].iloc[0]) is True
    assert frame["fdr"].iloc[0] == 2


def test_availability_zero_clears_minutes() -> None:
    frame = pd.DataFrame(
        {
            "status": ["i"],
            "chance_of_playing_next_round": [0],
            "played_r5": [1.0],
            "played_lag1": [1.0],
            "played_60_r5": [1.0],
            "minutes_lag1": [90.0],
            "minutes_r3": [90.0],
            "minutes_r5": [90.0],
            "has_fixture": [True],
        }
    )
    out = apply_fpl_availability(frame)
    assert out["played_r5"].iloc[0] == 0
    assert out["minutes_lag1"].iloc[0] == 0


def test_unavailable_players_are_dropped() -> None:
    frame = pd.DataFrame(
        {
            "status": ["u", "a"],
            "played_r5": [1.0, 1.0],
            "has_fixture": [True, True],
        }
    )
    out = apply_fpl_availability(frame)
    assert len(out) == 1


def test_unmapped_signing_gets_weak_minutes_not_fake_xg() -> None:
    panel = pd.DataFrame([_gw(10, 38, 90)])
    players = pd.DataFrame(
        [
            {
                "element_id": 50,
                "code": 9999,
                "web_name": "NewBoy",
                "team_id": 1,
                "position": "FWD",
                "now_cost": 70,
                "cost_m": 7.0,
                "status": "a",
                "chance_of_playing_next_round": 80,
            }
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "event": 1,
                "finished": False,
                "kickoff_time": "2026-08-15T14:00:00Z",
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
            }
        ]
    )
    code_map = pd.DataFrame({"prior_element_id": [10], "code": [1000]})
    frame, stats = build_live_feature_frame(
        panel,
        players,
        fixtures,
        season="2026-27",
        event=1,
        form_season="2025-26",
        map_by="code",
        code_map=code_map,
    )
    assert stats["n_mapped"] == 0
    assert stats["n_unmapped"] == 1
    assert frame["played_r5"].iloc[0] == 0.8
    assert pd.isna(frame["expected_goals_r5"].iloc[0])


def test_xmins_override_writes_roll_columns() -> None:
    frame = pd.DataFrame(
        {
            "element_id": [7],
            "minutes_lag1": [90.0],
            "minutes_r3": [90.0],
            "minutes_r5": [90.0],
            "played_r5": [1.0],
            "played_lag1": [1.0],
            "played_60_r5": [1.0],
        }
    )
    overrides = pd.DataFrame(
        {"element_id": [7], "event": [1], "p_play": [0.0], "e_minutes": [0.0], "p_60": [0.0]}
    )
    out = apply_xmins_to_features(frame, overrides, event=1)
    assert out["minutes_lag1"].iloc[0] == 0
    assert out["played_r5"].iloc[0] == 0
    assert bool(out["override"].iloc[0]) is True


def test_next_unfinished_event_skips_finished() -> None:
    fixtures = pd.DataFrame(
        {
            "event": [1, 1, 2],
            "finished": [True, True, False],
        }
    )
    assert next_unfinished_event(fixtures) == 2


def test_blank_gw_zeros_minutes() -> None:
    players = pd.DataFrame(
        {
            "element_id": [1],
            "team_id": [9],
            "web_name": ["Blank"],
            "position": ["DEF"],
            "status": ["a"],
            "played_r5": [1.0],
            "minutes_lag1": [90.0],
            "minutes_r3": [90.0],
            "minutes_r5": [90.0],
            "played_lag1": [1.0],
            "played_60_r5": [1.0],
        }
    )
    fixtures = pd.DataFrame(
        {
            "fixture_id": [1],
            "event": [1],
            "finished": [False],
            "team_h": [1],
            "team_a": [2],
            "team_h_difficulty": [3],
            "team_a_difficulty": [3],
            "kickoff_time": ["2026-08-15T14:00:00Z"],
        }
    )
    attached = attach_event_fixtures(players, fixtures, 1)
    assert bool(attached["has_fixture"].iloc[0]) is False
    out = apply_fpl_availability(attached)
    assert out["minutes_lag1"].iloc[0] == 0


def test_load_prediction_pool_uses_live_when_current_season_empty(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.vaastav_dir.mkdir(parents=True, exist_ok=True)
    (settings.vaastav_dir / "2025-26").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [10], "code": [1000]}).to_csv(
        settings.vaastav_dir / "2025-26" / "players_raw.csv", index=False
    )
    panel = pd.DataFrame([_gw(10, 38, 90)])
    players = pd.DataFrame(
        [
            {
                "element_id": 99,
                "code": 1000,
                "web_name": "Salah",
                "team_id": 1,
                "position": "MID",
                "now_cost": 145,
                "cost_m": 14.5,
                "status": "a",
                "chance_of_playing_next_round": 100,
            }
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "event": 1,
                "finished": False,
                "kickoff_time": "2026-08-15T14:00:00Z",
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            }
        ]
    )
    teams = pd.DataFrame({"team_id": [1, 2], "name": ["ARS", "AVL"], "short_name": ["ARS", "AVL"]})
    players.to_parquet(settings.processed_dir / "players.parquet", index=False)
    fixtures.to_parquet(settings.processed_dir / "fixtures.parquet", index=False)
    teams.to_parquet(settings.processed_dir / "teams.parquet", index=False)
    loaded = load_prediction_pool(panel, settings=settings, season="2026-27", event=1)
    assert loaded.source == "live_prior"
    assert loaded.event == 1
    assert loaded.n_mapped == 1
    assert len(loaded.pool) == 1
    assert loaded.pool["xpts"].iloc[0] > 0


def test_panel_has_gameweek() -> None:
    panel = pd.DataFrame([_gw(1, 38, 90)])
    assert panel_has_gameweek(panel, "2025-26", 38)
    assert not panel_has_gameweek(panel, "2026-27", 1)
