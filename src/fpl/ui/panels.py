"""My Team insight panels: TAKE/HOLD and chip EV as cards, not a terminal dump."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from fpl.models.horizon import merge_context
from fpl.optimize.chips import ChipReport
from fpl.optimize.squad import SquadSolution
from fpl.optimize.transfers import TransferPlan
from fpl.ui.hover import (
    _cell,
    catalog_row,
    merge_catalog,
    name_cell,
    render_html_table,
    render_player_frame,
)


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
        ("xG", "e_goals"),
        ("xA", "e_assists"),
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


def _player_side_cells(row: dict, catalog: pd.DataFrame | None) -> list[str]:
    data = merge_context(dict(row), catalog_row(catalog, row.get("element_id")))
    nxt = data.get("next_5_short") or data.get("next_5_text")
    news = str(data.get("news") or "").strip()
    if news in {"", "nan", "None", "<NA>"}:
        news_cell = "—"
    else:
        news_cell = html.escape(news if len(news) <= 42 else news[:40] + "…")
    return [
        name_cell(data, display=str(data.get("name") or "—")),
        _cell(data.get("team")),
        _cell(data.get("cost_m"), money=True),
        _cell(data.get("xpts"), digits=2),
        _cell(data.get("e_goals"), digits=2),
        _cell(data.get("e_assists"), digits=2),
        _cell(data.get("p_play"), digits=2),
        _cell(data.get("form"), digits=1),
        _cell(nxt),
        news_cell,
    ]


def render_moves_table(plan: TransferPlan, catalog: pd.DataFrame | None = None) -> None:
    n = max(len(plan.transfers_out), len(plan.transfers_in))
    if n <= 0:
        return
    rows: list[list[str]] = []
    for i in range(n):
        left = plan.transfers_out[i] if i < len(plan.transfers_out) else {}
        right = plan.transfers_in[i] if i < len(plan.transfers_in) else {}
        rows.append(_player_side_cells(left, catalog) + _player_side_cells(right, catalog))
    render_html_table(
        [
            "Out",
            "Club",
            "£m",
            "xPts",
            "xG",
            "xA",
            "p_play",
            "Form",
            "Next 5",
            "News",
            "In",
            "Club",
            "£m",
            "xPts",
            "xG",
            "xA",
            "p_play",
            "Form",
            "Next 5",
            "News",
        ],
        rows,
    )
    st.caption(
        "Incoming player gets the same columns as the sale: club, value, expected points this week, "
        "FPL expected goals / assists (xG / xA) scaled to expected minutes, chance they play, "
        "FPL form, next 5 (official FDR, 5=hardest), and official FPL news. "
        "Hover a name for this-GW transfer counts."
    )


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
    alts = list(getattr(plan, "alternatives", None) or [])
    if alts:
        st.markdown("**Other ideas** — pick **one** of these instead of the package")
        st.caption(
            "A two-move TAKE is a bundle. If you only like one of them, or the bank is tight, "
            "use a single below. If a row says it does not fit on its own, the other sale is "
            "what funds it — you cannot take that half by itself."
        )
        html_rows: list[list[str]] = []
        for alt in alts:
            outs = [
                merge_context(dict(r), catalog_row(catalog, r.get("element_id")))
                for r in alt.get("transfers_out") or []
            ]
            ins = [
                merge_context(dict(r), catalog_row(catalog, r.get("element_id")))
                for r in alt.get("transfers_in") or []
            ]
            legal = bool(alt.get("legal", True))
            in_next = ", ".join(
                str(r.get("next_5_short") or r.get("next_5_text") or "")
                for r in ins
            ).strip(", ")
            html_rows.append(
                [
                    html.escape(str(alt.get("label") or "Option")),
                    " ".join(name_cell(r) for r in outs) or "—",
                    " ".join(name_cell(r) for r in ins) or "—",
                    html.escape(in_next or "—"),
                    html.escape(f"{float(alt.get('expected_net') or 0):+.2f}"),
                    html.escape("yes" if legal else "no — not on its own"),
                    html.escape(str(int(alt.get("hits") or 0))),
                ]
            )
        render_html_table(
            ["Idea", "Out", "In", "In next 5", "vs hold", "Fits?", "Hits"],
            html_rows,
        )
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
