"""My Team insight panels: TAKE/HOLD and chip EV as cards, not a terminal dump."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.models.horizon import merge_context
from fpl.optimize.chips import ChipReport
from fpl.optimize.squad import SquadSolution
from fpl.optimize.transfers import TransferPlan
from fpl.ui.hover import catalog_row, merge_catalog, name_cell, render_html_table, render_player_frame


def _captain_name(solution: SquadSolution) -> str:
    rows = solution.table.loc[solution.table["is_captain"]]
    if rows.empty:
        return "?"
    return str(rows.iloc[0]["name"])


def xi_table(solution: SquadSolution, catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    table = solution.table.copy()
    table["role"] = "Bench"
    table.loc[table["in_xi"], "role"] = "XI"
    table.loc[table["is_captain"], "role"] = "Captain"
    table.loc[table["is_vice"], "role"] = "Vice"
    return merge_catalog(table, catalog)


def _xi_columns(table: pd.DataFrame) -> list[tuple[str, str]]:
    wanted = [
        ("Role", "role"),
        ("Pos", "position"),
        ("Player", "name"),
        ("Club", "team"),
        ("£m", "cost_m"),
        ("xPts", "xpts"),
        ("p_play", "p_play"),
        ("Form", "form"),
        ("Next 5", "next_5_short"),
    ]
    return [(h, c) for h, c in wanted if c in table.columns]


def render_xi_table(solution: SquadSolution, catalog: pd.DataFrame | None = None) -> None:
    table = xi_table(solution, catalog)
    cols = _xi_columns(table)
    if not cols:
        return
    render_player_frame(table, cols, name_keys={"name"})


def render_moves_table(plan: TransferPlan, catalog: pd.DataFrame | None = None) -> None:
    n = max(len(plan.transfers_out), len(plan.transfers_in))
    if n <= 0:
        return
    rows: list[list[str]] = []
    for i in range(n):
        left = plan.transfers_out[i] if i < len(plan.transfers_out) else {}
        right = plan.transfers_in[i] if i < len(plan.transfers_in) else {}
        left_d = merge_context(dict(left), catalog_row(catalog, left.get("element_id")))
        right_d = merge_context(dict(right), catalog_row(catalog, right.get("element_id")))
        lx = left.get("xpts")
        rx = right.get("xpts")
        lx_s = f"{float(lx):.2f}" if lx is not None and str(lx) != "" else "—"
        rx_s = f"{float(rx):.2f}" if rx is not None and str(rx) != "" else "—"
        rows.append(
            [
                name_cell(left_d, display=str(left_d.get("name") or "—")),
                lx_s,
                name_cell(right_d, display=str(right_d.get("name") or "—")),
                rx_s,
            ]
        )
    render_html_table(["Out", "Out xPts", "In", "In xPts"], rows)
    st.caption("Hover a name for club, value, FPL form, and next 5 (official FDR, 5=hardest).")


def render_transfer_plan(
    plan: TransferPlan,
    *,
    note: str | None = None,
    catalog: pd.DataFrame | None = None,
) -> None:
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
    if plan.transfers_out or plan.transfers_in:
        st.markdown("**Out → in**")
        render_moves_table(plan, catalog)
    elif verdict == "HOLD":
        st.caption("No move beats doing nothing after 4-point hits.")
    st.markdown("**Suggested XI**")
    render_xi_table(plan.chosen, catalog)


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


def render_squad_solution(
    solution: SquadSolution,
    *,
    note: str | None = None,
    catalog: pd.DataFrame | None = None,
) -> None:
    if note:
        st.caption(note)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XI EV", f"{solution.objective:.1f}")
    c2.metric("If-plays", f"{solution.naive_objective:.1f}", f"haircut {solution.haircut:.1f}")
    c3.metric("Bank", f"£{solution.bank_m:.1f}m")
    c4.metric("Captain", _captain_name(solution))
    st.caption("Wildcard-shaped 15 — ignores your current squad. Comparison only.")
    render_xi_table(solution, catalog)
