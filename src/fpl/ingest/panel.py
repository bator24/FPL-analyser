from __future__ import annotations

from pathlib import Path

import pandas as pd

POSITION_MAP = {
    "GK": "GKP",
    "GKP": "GKP",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

OUTCOME_COLUMNS = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def normalize_merged_gw(path: Path, season: str, teams_path: Path | None = None) -> pd.DataFrame:
    raw = _read_csv(path)
    n = len(raw)

    def col(name: str, default=pd.NA):
        if name in raw.columns:
            return raw[name]
        return pd.Series([default] * n, index=raw.index)

    if "round" in raw.columns:
        event = raw["round"]
    elif "GW" in raw.columns:
        event = raw["GW"]
    else:
        raise ValueError(f"{path} has no round/GW column")

    position = col("position")
    position = position.astype("string").str.upper().map(POSITION_MAP).fillna(position)

    data = {
        "season": season,
        "element_id": col("element"),
        "event": event,
        "name": col("name"),
        "position": position,
        "team": col("team"),
        "opponent_team": col("opponent_team"),
        "fixture_id": col("fixture"),
        "was_home": col("was_home"),
        "kickoff_time": col("kickoff_time"),
        "team_h_score": col("team_h_score"),
        "team_a_score": col("team_a_score"),
        "value": col("value"),
        "selected": col("selected"),
        "source": "vaastav",
        "fpl_xp_posthoc": col("xP"),
    }
    for column in OUTCOME_COLUMNS:
        data[column] = col(column)
    frame = pd.DataFrame(data)

    numeric_cols = [
        "element_id",
        "event",
        "opponent_team",
        "fixture_id",
        "value",
        "selected",
        "fpl_xp_posthoc",
        "team_h_score",
        "team_a_score",
        *OUTCOME_COLUMNS,
    ]
    frame = _to_numeric(frame, numeric_cols)
    if "was_home" in frame.columns:
        frame["was_home"] = frame["was_home"].astype("boolean")
    frame["cost_m"] = frame["value"] / 10.0
    frame["played"] = (frame["minutes"].fillna(0) > 0).astype(int)
    frame["played_60"] = (frame["minutes"].fillna(0) >= 60).astype(int)

    if teams_path is not None and teams_path.exists():
        teams = _read_csv(teams_path)
        name_col = "name" if "name" in teams.columns else None
        if name_col and "id" in teams.columns:
            team_map = teams.rename(columns={"id": "team_id", "name": "team"})[["team_id", "team"]]
            frame = frame.merge(team_map, on="team", how="left")
        short = teams.rename(columns={"id": "opponent_team", "short_name": "opponent_short"})
        if "opponent_short" in short.columns:
            frame = frame.merge(short[["opponent_team", "opponent_short"]], on="opponent_team", how="left")
    if "team_id" not in frame.columns:
        frame["team_id"] = pd.NA
    return frame


def attach_fdr(player_gw: pd.DataFrame, fixtures_path: Path | None) -> pd.DataFrame:
    if fixtures_path is None or not fixtures_path.exists() or player_gw.empty:
        out = player_gw.copy()
        if "fdr" not in out.columns:
            out["fdr"] = pd.NA
        return out
    fixtures = _read_csv(fixtures_path)
    keep = [c for c in ["id", "team_h_difficulty", "team_a_difficulty"] if c in fixtures.columns]
    if "id" not in keep:
        out = player_gw.copy()
        out["fdr"] = pd.NA
        return out
    fixtures = fixtures[keep].rename(columns={"id": "fixture_id"})
    merged = player_gw.merge(fixtures, on="fixture_id", how="left")
    home = merged["was_home"].fillna(False).astype(bool)
    merged["fdr"] = merged["team_h_difficulty"].where(home, merged["team_a_difficulty"])
    return merged.drop(columns=["team_h_difficulty", "team_a_difficulty"], errors="ignore")


def normalize_element_histories(
    summaries: dict[int, dict],
    players: pd.DataFrame,
    teams: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """Turn live element-summary history into the same player-GW schema."""
    rows: list[dict] = []
    player_lookup = players.set_index("element_id", drop=False)
    team_names = {}
    if not teams.empty and "team_id" in teams.columns:
        team_names = teams.set_index("team_id")["name"].to_dict() if "name" in teams.columns else {}
        team_short = (
            teams.set_index("team_id")["short_name"].to_dict() if "short_name" in teams.columns else {}
        )
    else:
        team_short = {}

    for element_id, payload in summaries.items():
        history = payload.get("history") or []
        meta = player_lookup.loc[element_id] if element_id in player_lookup.index else None
        for item in history:
            team_id = int(meta["team_id"]) if meta is not None and pd.notna(meta.get("team_id")) else None
            rows.append(
                {
                    "season": season,
                    "element_id": element_id,
                    "event": item.get("round"),
                    "name": None if meta is None else meta.get("web_name"),
                    "position": None if meta is None else meta.get("position"),
                    "team": team_names.get(team_id) if team_id is not None else None,
                    "team_id": team_id,
                    "opponent_team": item.get("opponent_team"),
                    "opponent_short": team_short.get(item.get("opponent_team")),
                    "fixture_id": item.get("fixture"),
                    "was_home": item.get("was_home"),
                    "kickoff_time": item.get("kickoff_time"),
                    "team_h_score": item.get("team_h_score"),
                    "team_a_score": item.get("team_a_score"),
                    "value": item.get("value"),
                    "selected": item.get("selected"),
                    "source": "element-summary",
                    "fpl_xp_posthoc": pd.NA,
                    **{column: item.get(column) for column in OUTCOME_COLUMNS},
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = _to_numeric(frame, ["element_id", "event", "opponent_team", "fixture_id", "value", "selected", *OUTCOME_COLUMNS])
    frame["was_home"] = frame["was_home"].astype("boolean")
    frame["cost_m"] = frame["value"] / 10.0
    frame["played"] = (frame["minutes"].fillna(0) > 0).astype(int)
    frame["played_60"] = (frame["minutes"].fillna(0) >= 60).astype(int)
    return frame


def add_match_goals(player_gw: pd.DataFrame) -> pd.DataFrame:
    """Team/opponent goals from the FPL scoreline on each player row."""
    out = player_gw.copy()
    if "team_h_score" in out.columns and "team_a_score" in out.columns:
        home = out["was_home"].fillna(False).astype(bool)
        out["team_goals"] = out["team_h_score"].where(home, out["team_a_score"])
        out["opp_goals"] = out["team_a_score"].where(home, out["team_h_score"])
        return out
    # vaastav merged_gw has team_h_score / team_a_score; if missing, leave empty
    if "team_goals" not in out.columns:
        out["team_goals"] = pd.NA
        out["opp_goals"] = pd.NA
    return out
