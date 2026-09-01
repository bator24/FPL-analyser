from __future__ import annotations

from typing import Any

import pandas as pd

POSITION_SHORT = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

_PLAYER_NUMERIC = [
    "now_cost",
    "selected_by_percent",
    "chance_of_playing_this_round",
    "chance_of_playing_next_round",
    "ep_this",
    "ep_next",
    "form",
    "points_per_game",
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "event_points",
    "transfers_in",
    "transfers_out",
    "transfers_in_event",
    "transfers_out_event",
]


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _pick(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def normalize(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]] | dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Turn raw FPL JSON into stable tables."""
    fixture_rows = fixtures if isinstance(fixtures, list) else fixtures.get("fixtures", [])

    teams = _pick(
        list(bootstrap.get("teams") or []),
        [
            "id",
            "name",
            "short_name",
            "code",
            "strength",
            "strength_overall_home",
            "strength_overall_away",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
            "pulse_id",
        ],
    )
    teams = teams.rename(columns={"id": "team_id"})

    element_types = _pick(
        list(bootstrap.get("element_types") or []),
        ["id", "singular_name", "singular_name_short", "squad_select", "squad_min_play", "squad_max_play"],
    )
    element_types = element_types.rename(columns={"id": "element_type"})

    events = _pick(
        list(bootstrap.get("events") or []),
        [
            "id",
            "name",
            "deadline_time",
            "average_entry_score",
            "finished",
            "data_checked",
            "is_previous",
            "is_current",
            "is_next",
            "highest_score",
        ],
    )
    events = events.rename(columns={"id": "event"})

    players = _pick(
        list(bootstrap.get("elements") or []),
        [
            "id",
            "code",
            "first_name",
            "second_name",
            "web_name",
            "team",
            "team_code",
            "element_type",
            "now_cost",
            "selected_by_percent",
            "status",
            "news",
            "news_added",
            "chance_of_playing_this_round",
            "chance_of_playing_next_round",
            "ep_this",
            "ep_next",
            "form",
            "points_per_game",
            "total_points",
            "minutes",
            "starts",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
            "event_points",
            "transfers_in",
            "transfers_out",
            "transfers_in_event",
            "transfers_out_event",
        ],
    )
    players = _numeric(players, _PLAYER_NUMERIC)
    players = players.rename(columns={"id": "element_id", "team": "team_id"})
    players["position"] = players["element_type"].map(POSITION_SHORT)
    players["cost_m"] = players["now_cost"] / 10.0

    fixture_table = _pick(
        list(fixture_rows),
        [
            "id",
            "code",
            "event",
            "finished",
            "kickoff_time",
            "minutes",
            "started",
            "team_a",
            "team_h",
            "team_a_score",
            "team_h_score",
            "team_h_difficulty",
            "team_a_difficulty",
            "pulse_id",
        ],
    )
    fixture_table = fixture_table.rename(columns={"id": "fixture_id"})

    return {
        "teams": teams,
        "element_types": element_types,
        "events": events,
        "players": players,
        "fixtures": fixture_table,
    }
