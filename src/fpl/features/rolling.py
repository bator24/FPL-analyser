from __future__ import annotations

import pandas as pd

ROLL_WINDOWS = (1, 3, 5, 10, 38)

PLAYER_ROLL_COLS = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "ict_index",
    "played",
    "played_60",
]

TEAM_ROLL_COLS = ["team_goals", "opp_goals"]


def _shifted_rolling_mean(lagged: pd.Series, season: pd.Series, element_id: pd.Series, window: int) -> pd.Series:
    rolled = lagged.groupby([season, element_id], sort=False).rolling(window, min_periods=1).mean()
    rolled = rolled.reset_index(level=[0, 1], drop=True)
    return rolled.reindex(lagged.index)


def add_player_rolling(player_gw: pd.DataFrame, windows: tuple[int, ...] = ROLL_WINDOWS) -> pd.DataFrame:
    """Rolling means of prior matches only. Current-row outcomes are never in the feature."""
    out = player_gw.copy()
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")
    out = out.sort_values(["season", "element_id", "kickoff_time", "fixture_id"], kind="mergesort")
    grouped = out.groupby(["season", "element_id"], sort=False)
    for column in PLAYER_ROLL_COLS:
        if column not in out.columns:
            continue
        lagged = grouped[column].shift(1)
        out[f"{column}_lag1"] = lagged
        for window in windows:
            out[f"{column}_r{window}"] = _shifted_rolling_mean(
                lagged, out["season"], out["element_id"], window
            )
    return out


def add_team_opponent_rolling(player_gw: pd.DataFrame, windows: tuple[int, ...] = ROLL_WINDOWS) -> pd.DataFrame:
    """Team attack/defence form, shifted, then mapped onto the opponent as well."""
    out = player_gw.copy()
    if "team_id" not in out.columns or out["team_id"].isna().all():
        return out
    if "team_goals" not in out.columns:
        return out

    team_matches = (
        out.dropna(subset=["team_id", "fixture_id"])
        .sort_values(["season", "team_id", "kickoff_time", "fixture_id"], kind="mergesort")
        .drop_duplicates(["season", "team_id", "fixture_id"], keep="first")
        [["season", "team_id", "fixture_id", "kickoff_time", "team_goals", "opp_goals"]]
        .copy()
    )
    grouped = team_matches.groupby(["season", "team_id"], sort=False)
    feat_cols: list[str] = []
    for column in TEAM_ROLL_COLS:
        lagged = grouped[column].shift(1)
        for window in windows:
            name = f"{column}_r{window}"
            team_matches[name] = _shifted_rolling_mean(
                lagged, team_matches["season"], team_matches["team_id"], window
            )
            feat_cols.append(name)

    team_key = team_matches[["season", "team_id", "fixture_id", *feat_cols]]
    out = out.merge(team_key, on=["season", "team_id", "fixture_id"], how="left")

    opp_key = team_key.rename(
        columns={
            "team_id": "opponent_team",
            **{f"team_goals_r{w}": f"opp_gf_r{w}" for w in windows},
            **{f"opp_goals_r{w}": f"opp_ga_r{w}" for w in windows},
        }
    )
    return out.merge(opp_key, on=["season", "opponent_team", "fixture_id"], how="left")


def build_asof_features(player_gw: pd.DataFrame) -> pd.DataFrame:
    framed = add_player_rolling(player_gw)
    return add_team_opponent_rolling(framed)
