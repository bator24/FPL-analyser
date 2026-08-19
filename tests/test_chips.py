from fpl.optimize.chips import bb_ev, bench_xpts, evaluate_chips, tc_ev
from fpl.optimize.transfers import solve_transfers
from tests.test_squad import toy_pool
from tests.test_transfers import toy_market


def test_bb_gain_is_bench_xpts() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, free_transfers=0, include_wildcard=False)
    hold = plan.hold
    assert abs(bb_ev(hold) - hold.objective - bench_xpts(hold)) < 1e-9
    assert bb_ev(hold) > hold.objective


def test_tc_gain_is_one_extra_captain_copy() -> None:
    hold = solve_transfers(toy_pool(), set(toy_pool()["element_id"]), include_wildcard=False).hold
    cap = float(hold.table.loc[hold.table["is_captain"], "xpts"].iloc[0])
    assert abs(tc_ev(hold) - hold.objective - cap) < 1e-9
    assert hold.table.loc[hold.table["is_captain"], "name"].iloc[0] == "Nailed"


def test_fh_matches_wildcard_this_gw() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, include_wildcard=True)
    report = evaluate_chips(plan)
    fh = report.option("fh_wc")
    assert plan.wildcard is not None
    assert abs(fh.ev - plan.wildcard.objective) < 1e-9
    assert fh.gain_vs_hold > 0


def test_chip_report_ranks_rebuild_and_keeps_hold_baseline() -> None:
    pool, current = toy_market(9.0)
    plan = solve_transfers(pool, current, include_wildcard=True)
    report = evaluate_chips(plan)
    assert report.option("hold").gain_vs_hold == 0
    assert report.option("tc_hold").gain_vs_hold > 0
    assert report.option("bb_hold").gain_vs_hold > 0
    assert report.best_this_gw in {row.key for row in report.options}
