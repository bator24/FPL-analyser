import pandas as pd

from fpl.optimize.transfers import solve_transfers
from tests.test_squad import _player, toy_pool


def toy_market(upgrade_xpts: float) -> tuple[pd.DataFrame, set[int]]:
    base = toy_pool()
    extra = _player(
        304,
        "FWD",
        name="Upgrade",
        team_id=14,
        cost_m=8.0,
        xpts=upgrade_xpts,
        p_play=1.0,
    )
    return pd.concat([base, pd.DataFrame([extra])], ignore_index=True), set(base["element_id"])


def test_free_transfer_takes_small_upgrade() -> None:
    pool, current = toy_market(5.0)
    plan = solve_transfers(pool, current, free_transfers=1, include_wildcard=False)
    names_in = {row["name"] for row in plan.transfers_in}
    names_out = {row["name"] for row in plan.transfers_out}
    assert plan.n_transfers == 1
    assert plan.hits == 0
    assert "Upgrade" in names_in
    assert "Cheap" in names_out
    assert plan.expected_net > 0
    assert plan.recommend


def test_hit_refused_when_gain_below_four() -> None:
    pool, current = toy_market(5.0)
    plan = solve_transfers(pool, current, free_transfers=0, include_wildcard=False)
    assert plan.n_transfers == 0
    assert plan.hits == 0
    assert not plan.recommend
    assert plan.expected_net > -0.05


def test_hit_taken_when_gain_beats_four() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, free_transfers=0, include_wildcard=False)
    assert plan.hits == 1
    assert plan.n_transfers == 1
    assert {row["name"] for row in plan.transfers_in} == {"Upgrade"}
    assert plan.expected_net > 0
    assert plan.recommend
    assert plan.chosen.legal


def test_hold_is_legal_xi_on_current_fifteen() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, free_transfers=0, include_wildcard=False)
    assert plan.hold.legal
    assert plan.hold.squad_ids == current
    assert int(plan.hold.table["in_xi"].sum()) == 11


def test_locked_squad_solves_when_none_pass_captain_gate() -> None:
    """Hold must not require a captain who is not in the 15."""
    current = toy_pool()
    current["p_play"] = 0.2
    market = toy_market(9.0)[0]
    market.loc[market["element_id"].isin(current["element_id"]), "p_play"] = 0.2
    plan = solve_transfers(market, set(current["element_id"]), include_wildcard=False)
    assert plan.hold.legal
    assert plan.hold.squad_ids == set(current["element_id"])


def test_wildcard_rebuilds_without_hits() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, wildcard=True, include_wildcard=True)
    assert plan.mode == "wildcard"
    assert plan.chosen.hits == 0
    assert "Upgrade" in set(plan.chosen.table["name"])
    assert plan.chosen.legal
