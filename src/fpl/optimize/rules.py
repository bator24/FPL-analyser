"""FPL squad construction rules (Ramezani/Matthews constraint pattern)."""

from __future__ import annotations

from typing import Any

import pandas as pd

SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
POSITION_ORDER = ("GKP", "DEF", "MID", "FWD")
BUDGET_M = 100.0
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
XI_SIZE = 11
CAPTAIN_MIN_P_PLAY = 0.75
MAX_XPTS_IF_PLAYS = 20.0
HIT_COST = 4.0
FREE_TRANSFERS_DEFAULT = 1
MAX_TRANSFERS_DEFAULT = 3


def normalize_position(value: Any) -> str:
    pos = str(value).strip().upper()
    aliases = {
        "GK": "GKP",
        "GKP": "GKP",
        "DEF": "DEF",
        "MID": "MID",
        "FWD": "FWD",
        "AM": "MID",
        "DM": "MID",
        "CM": "MID",
        "WM": "MID",
        "WB": "DEF",
        "CB": "DEF",
        "LB": "DEF",
        "RB": "DEF",
        "ST": "FWD",
        "CF": "FWD",
    }
    if pos in aliases:
        return aliases[pos]
    if pos in POSITION_ORDER:
        return pos
    raise ValueError(f"Unknown position: {value!r}")


def club_key(frame: pd.DataFrame) -> pd.Series:
    if "team_id" in frame.columns and frame["team_id"].notna().any():
        team_id = pd.to_numeric(frame["team_id"], errors="coerce")
        if team_id.notna().any():
            return team_id.fillna(-frame.index.to_series() - 1).astype(int).astype("string")
    if "team" in frame.columns:
        return frame["team"].astype("string").fillna("unknown")
    return pd.Series(["unknown"] * len(frame), index=frame.index, dtype="string")


def formation_from_xi(xi: pd.DataFrame) -> str:
    counts = xi["position"].map(normalize_position).value_counts()
    return f"{int(counts.get('DEF', 0))}-{int(counts.get('MID', 0))}-{int(counts.get('FWD', 0))}"


def legality_errors(
    squad: pd.DataFrame,
    *,
    xi: pd.DataFrame | None = None,
    captain_id: int | None = None,
    vice_id: int | None = None,
    budget_m: float = BUDGET_M,
    max_per_club: int = MAX_PER_CLUB,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
) -> list[str]:
    """Return human-readable rule breaches; empty means the squad is legal."""
    errors: list[str] = []
    if squad.empty:
        return ["squad is empty"]
    table = squad.copy()
    table["position"] = table["position"].map(normalize_position)
    table["element_id"] = pd.to_numeric(table["element_id"], errors="coerce")
    if table["element_id"].duplicated().any():
        errors.append("duplicate element_id in squad")
    if len(table) != SQUAD_SIZE:
        errors.append(f"squad size {len(table)} != {SQUAD_SIZE}")
    counts = table["position"].value_counts()
    for pos, need in SQUAD_COUNTS.items():
        got = int(counts.get(pos, 0))
        if got != need:
            errors.append(f"squad {pos} {got} != {need}")
    spent = float(pd.to_numeric(table["cost_m"], errors="coerce").fillna(0).sum())
    if spent > budget_m + 1e-9:
        errors.append(f"spent {spent:.1f} > budget {budget_m:.1f}")
    clubs = club_key(table).value_counts()
    over = clubs[clubs > max_per_club]
    if not over.empty:
        errors.append(f"club cap broken: {over.to_dict()}")

    if xi is None:
        return errors
    xi_table = xi.copy()
    xi_table["position"] = xi_table["position"].map(normalize_position)
    xi_ids = set(pd.to_numeric(xi_table["element_id"], errors="coerce"))
    squad_ids = set(pd.to_numeric(table["element_id"], errors="coerce"))
    if not xi_ids.issubset(squad_ids):
        errors.append("XI contains players outside the squad")
    if len(xi_table) != XI_SIZE:
        errors.append(f"XI size {len(xi_table)} != {XI_SIZE}")
    xi_counts = xi_table["position"].value_counts()
    for pos, (lo, hi) in XI_BOUNDS.items():
        got = int(xi_counts.get(pos, 0))
        if got < lo or got > hi:
            errors.append(f"XI {pos} {got} not in [{lo}, {hi}]")
    if int(xi_counts.get("GKP", 0)) != 1:
        errors.append("XI must have exactly one GKP")

    if captain_id is not None:
        if captain_id not in xi_ids:
            errors.append("captain is not in the XI")
        cap_rows = xi_table[pd.to_numeric(xi_table["element_id"], errors="coerce") == captain_id]
        if not cap_rows.empty and "p_play" in cap_rows.columns:
            eligible_in_xi = xi_table[pd.to_numeric(xi_table["p_play"], errors="coerce").fillna(0) >= captain_min_p_play]
            cap_p = float(pd.to_numeric(cap_rows["p_play"], errors="coerce").iloc[0])
            if len(eligible_in_xi) and cap_p < captain_min_p_play:
                errors.append(
                    f"captain p_play {cap_p:.2f} below gate {captain_min_p_play:.2f} with eligible alternatives"
                )
    if vice_id is not None:
        if vice_id not in xi_ids:
            errors.append("vice is not in the XI")
        if vice_id == captain_id:
            errors.append("vice equals captain")
    return errors


def is_legal(*args: Any, **kwargs: Any) -> bool:
    return not legality_errors(*args, **kwargs)


def xi_shape_options() -> tuple[tuple[int, int, int], ...]:
    shapes = []
    for n_def in range(XI_BOUNDS["DEF"][0], XI_BOUNDS["DEF"][1] + 1):
        for n_mid in range(XI_BOUNDS["MID"][0], XI_BOUNDS["MID"][1] + 1):
            for n_fwd in range(XI_BOUNDS["FWD"][0], XI_BOUNDS["FWD"][1] + 1):
                if n_def + n_mid + n_fwd == 10:
                    shapes.append((n_def, n_mid, n_fwd))
    return tuple(shapes)
