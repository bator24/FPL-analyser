import pandas as pd

from fpl.optimize.rules import legality_errors
from fpl.optimize.squad import brute_force_squad, solve_squad


def _player(
    element_id: int,
    position: str,
    *,
    name: str | None = None,
    team_id: int,
    cost_m: float = 4.5,
    xpts: float,
    p_play: float = 0.9,
    xpts_if_plays: float | None = None,
) -> dict:
    if_plays = xpts_if_plays if xpts_if_plays is not None else (xpts / p_play if p_play else xpts)
    return {
        "element_id": element_id,
        "name": name or f"{position}-{element_id}",
        "position": position,
        "team_id": team_id,
        "team": f"T{team_id}",
        "cost_m": cost_m,
        "xpts": xpts,
        "p_play": p_play,
        "xpts_if_plays": if_plays,
    }


def toy_pool() -> pd.DataFrame:
    """15 players. Flip has a huge if-he-plays mean but fails the captain gate."""
    rows = []
    # 2 GKP, 5 DEF, 5 MID — nailed, cheap, unique clubs.
    xpts = 2.0
    eid = 1
    for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5)]:
        for _ in range(n):
            rows.append(_player(eid, pos, team_id=eid, cost_m=4.5, xpts=xpts, p_play=0.92))
            xpts += 0.15
            eid += 1
    rows.append(
        _player(
            301,
            "FWD",
            name="Flip",
            team_id=11,
            cost_m=8.0,
            xpts=4.8,
            p_play=0.40,
            xpts_if_plays=12.0,
        )
    )
    rows.append(
        _player(
            302,
            "FWD",
            name="Nailed",
            team_id=12,
            cost_m=8.0,
            xpts=6.175,
            p_play=0.95,
            xpts_if_plays=6.5,
        )
    )
    rows.append(
        _player(
            303,
            "FWD",
            name="Cheap",
            team_id=13,
            cost_m=5.5,
            xpts=3.6,
            p_play=0.90,
            xpts_if_plays=4.0,
        )
    )
    return pd.DataFrame(rows)


def test_illegal_when_four_from_one_club() -> None:
    pool = toy_pool()
    pool.loc[pool["element_id"].isin([1, 2, 3, 4]), "team_id"] = 1
    errors = legality_errors(pool)
    assert any("club cap" in e for e in errors)


def test_solver_matches_brute_force_on_toy() -> None:
    pool = toy_pool()
    pulp_sol = solve_squad(pool, season="toy", event=1)
    brute = brute_force_squad(pool, season="toy", event=1)
    assert pulp_sol.legal
    assert brute.legal
    pulp_ids = set(pulp_sol.table["element_id"])
    brute_ids = set(brute.table["element_id"])
    assert pulp_ids == brute_ids
    assert set(pulp_sol.xi["element_id"]) == set(brute.xi["element_id"])
    assert pulp_sol.captain_id == brute.captain_id
    assert abs(pulp_sol.objective - brute.objective) < 1e-6


def test_rotation_risk_loses_captain_to_nailed() -> None:
    pool = toy_pool()
    sol = solve_squad(pool)
    names = sol.table.set_index("element_id")["name"]
    assert names.at[sol.captain_id] == "Nailed"
    xi_names = set(sol.xi["name"])
    assert "Flip" in xi_names
    # Mean-only (ignore p_play gate) would armband Flip.
    xi = sol.xi
    naive_cap = xi.sort_values("xpts_if_plays", ascending=False).iloc[0]
    assert naive_cap["name"] == "Flip"
    assert naive_cap["p_play"] < 0.75


def test_solution_always_reports_haircut() -> None:
    sol = solve_squad(toy_pool())
    assert sol.naive_objective > sol.objective
    assert sol.formation.count("-") == 2
    assert sol.bank_m >= -1e-9
    assert int(sol.table["in_xi"].sum()) == 11
    assert int((~sol.table["in_xi"]).sum()) == 4
