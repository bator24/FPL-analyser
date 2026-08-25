"""My Team insight panels: TAKE/HOLD and chip EV as cards, not a terminal dump."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.optimize.chips import ChipReport
from fpl.optimize.squad import SquadSolution
from fpl.optimize.transfers import TransferPlan


def _captain_name(solution: SquadSolution) -> str:
    rows = solution.table.loc[solution.table["is_captain"]]
    if rows.empty:
        return "?"
    return str(rows.iloc[0]["name"])


def xi_table(solution: SquadSolution) -> pd.DataFrame:
    table = solution.table.copy()
    table["role"] = "Bench"
    table.loc[table["in_xi"], "role"] = "XI"
    table.loc[table["is_captain"], "role"] = "Captain"
    table.loc[table["is_vice"], "role"] = "Vice"
    cols = [c for c in ["role", "position", "name", "cost_m", "xpts", "p_play"] if c in table.columns]
    return table[cols]


def moves_table(plan: TransferPlan) -> pd.DataFrame:
    n = max(len(plan.transfers_out), len(plan.transfers_in))
    rows = []
    for i in range(n):
        left = plan.transfers_out[i] if i < len(plan.transfers_out) else {}
        right = plan.transfers_in[i] if i < len(plan.transfers_in) else {}
        rows.append(
            {
                "Out": left.get("name", "—"),
                "Out xPts": left.get("xpts"),
                "In": right.get("name", "—"),
                "In xPts": right.get("xpts"),
            }
        )
    return pd.DataFrame(rows)


def render_transfer_plan(plan: TransferPlan, *, note: str | None = None) -> None:
    verdict = "TAKE" if plan.recommend else "HOLD"
    if plan.mode == "wildcard":
        verdict = "WILDCARD" if plan.recommend else "HOLD"
    klass = "fpl-verdict-take" if plan.recommend else "fpl-verdict-hold"
    st.markdown(f'<div class="fpl-verdict {klass}">{verdict}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hold EV", f"{plan.hold.objective:.1f}")
    m2.metric("Chosen net", f"{plan.chosen.net_objective:.1f}", f"{plan.expected_net:+.1f} vs hold")
    m3.metric("Moves / hits", f"{plan.n_transfers} / {plan.hits}")
    m4.metric("Captain", _captain_name(plan.chosen))
    moves = moves_table(plan)
    if not moves.empty:
        st.markdown("**Out → in**")
        st.dataframe(moves, hide_index=True, use_container_width=True)
    elif verdict == "HOLD":
        st.caption("No move beats doing nothing after 4-point hits.")
    st.markdown("**Suggested XI**")
    st.dataframe(xi_table(plan.chosen), hide_index=True, use_container_width=True)


def render_chip_report(report: ChipReport, *, note: str | None = None) -> None:
    if note:
        st.caption(note)
    best = next((row for row in report.options if row.key == report.best_this_gw), None)
    st.markdown(
        f'<div class="fpl-verdict fpl-verdict-hold">Best this GW: '
        f"{best.label if best else report.best_this_gw}</div>",
        unsafe_allow_html=True,
    )
    if report.chip_beats_no_chip:
        st.warning(
            "A chip wins *this week* on arithmetic. That is not an instruction to play it — "
            "saving TC for a nailed premium (especially a DGW) is usually right."
        )
    rows = [
        {
            "Option": row.label,
            "EV": round(row.ev, 2),
            "vs hold": round(row.gain_vs_hold, 2),
            "Best": "★" if row.key == report.best_this_gw else "",
        }
        for row in report.options
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_squad_solution(solution: SquadSolution, *, note: str | None = None) -> None:
    if note:
        st.caption(note)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XI EV", f"{solution.objective:.1f}")
    c2.metric("If-plays", f"{solution.naive_objective:.1f}", f"haircut {solution.haircut:.1f}")
    c3.metric("Bank", f"£{solution.bank_m:.1f}m")
    c4.metric("Captain", _captain_name(solution))
    st.caption("Wildcard-shaped 15 — ignores your current squad. Comparison only.")
    st.dataframe(xi_table(solution), hide_index=True, use_container_width=True)
