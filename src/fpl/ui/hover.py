"""Hover cards for player tables. Streamlit dataframes have no per-row tooltip."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

HOVER_CSS = """
<style>
.fpl-hover-wrap {
  overflow: visible;
  margin: 0.2rem 0 0.6rem 0;
}
.fpl-hover-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
  color: #e4e4e7;
}
.fpl-hover-table th {
  text-align: left;
  color: #a1a1aa;
  font-weight: 650;
  padding: 0.32rem 0.45rem;
  border-bottom: 1px solid #3f3f46;
  white-space: nowrap;
}
.fpl-hover-table td {
  padding: 0.32rem 0.45rem;
  border-bottom: 1px solid #27272a;
  vertical-align: top;
}
.fpl-name {
  position: relative;
  cursor: help;
  border-bottom: 1px dotted #a1a1aa;
  font-weight: 650;
  color: #fafafa;
}
.fpl-name .fpl-tip {
  display: none;
  position: absolute;
  left: 0;
  top: calc(100% + 6px);
  z-index: 40;
  min-width: 250px;
  max-width: 360px;
  background: #18181b;
  border: 1px solid #52525b;
  padding: 0.55rem 0.7rem;
  border-radius: 8px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.55);
  color: #f4f4f5;
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.4;
  white-space: pre-wrap;
}
.fpl-name:hover .fpl-tip,
.fpl-name:focus .fpl-tip {
  display: block;
}
</style>
"""

_CONTEXT_COLS = (
    "team",
    "form",
    "cost_m",
    "points",
    "this_gw",
    "next_5_short",
    "next_5_text",
    "fixture_verdict",
    "fdr_mean",
    "hard_n",
    "next_fdr",
    "points_per_game",
)


def inject_hover_css() -> None:
    if st.session_state.get("_fpl_hover_css"):
        return
    st.markdown(HOVER_CSS, unsafe_allow_html=True)
    st.session_state["_fpl_hover_css"] = True


def _plain(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in {"", "nan", "<NA>", "None"}:
        return ""
    return text


def _fmt_money(value: Any) -> str:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(n):
        return ""
    return f"£{float(n):.1f}m"


def _fmt_form(value: Any) -> str:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(n):
        return ""
    return f"{float(n):.1f}"


def tooltip_text(row: dict[str, Any] | pd.Series) -> str:
    """Plain-text card for native title= and Streamlit button help=."""
    data = dict(row) if not isinstance(row, dict) else row
    name = _plain(data.get("name")) or "?"
    club = _plain(data.get("team"))
    cost = _fmt_money(data.get("cost_m"))
    form = _fmt_form(data.get("form"))
    pts = _plain(data.get("points"))
    ppg = _fmt_form(data.get("points_per_game"))
    header = " · ".join(p for p in (name, club, cost) if p)
    bits = [header] if header else []
    form_bits = []
    if form:
        form_bits.append(f"FPL form {form}")
    if ppg:
        form_bits.append(f"PPG {ppg}")
    if pts:
        form_bits.append(f"{pts} pts")
    if form_bits:
        bits.append(" · ".join(form_bits))
    this_gw = _plain(data.get("this_gw"))
    if this_gw:
        bits.append(f"This GW: {this_gw}")
    nxt = _plain(data.get("next_5_text")) or _plain(data.get("next_5_short"))
    if nxt:
        bits.append(f"Next 5 (FPL FDR, 5=hardest): {nxt}")
    verdict = _plain(data.get("fixture_verdict"))
    if verdict:
        bits.append(verdict)
    return "\n".join(bits) if bits else name


def name_cell(row: dict[str, Any] | pd.Series, display: str | None = None) -> str:
    data = dict(row) if not isinstance(row, dict) else row
    label = html.escape(display or _plain(data.get("name")) or "—")
    tip = html.escape(tooltip_text(data))
    inner = tip.replace("\n", "<br/>")
    return (
        f'<span class="fpl-name" tabindex="0" title="{tip}">{label}'
        f'<span class="fpl-tip">{inner}</span></span>'
    )


def catalog_row(catalog: pd.DataFrame | None, element_id: Any) -> dict[str, Any]:
    if catalog is None or catalog.empty or element_id is None or "element_id" not in catalog.columns:
        return {}
    try:
        eid = int(element_id)
    except (TypeError, ValueError):
        return {}
    hit = catalog.loc[pd.to_numeric(catalog["element_id"], errors="coerce") == eid]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def merge_catalog(frame: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or catalog is None or catalog.empty:
        return frame
    if "element_id" not in frame.columns or "element_id" not in catalog.columns:
        return frame
    extra = [c for c in _CONTEXT_COLS if c in catalog.columns]
    if not extra:
        return frame
    src = catalog[["element_id", *extra]].drop_duplicates("element_id")
    out = frame.copy()
    out["element_id"] = pd.to_numeric(out["element_id"], errors="coerce")
    src = src.copy()
    src["element_id"] = pd.to_numeric(src["element_id"], errors="coerce")
    merged = out.merge(src, on="element_id", how="left", suffixes=("", "_cat"))
    for col in extra:
        cat = f"{col}_cat"
        if cat in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[cat])
            merged = merged.drop(columns=[cat])
    return merged


def render_html_table(headers: list[str], rows: list[list[str]], *, max_height: int | None = None) -> None:
    inject_hover_css()
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    style = f"max-height:{int(max_height)}px;overflow:auto;" if max_height else ""
    st.markdown(
        f'<div class="fpl-hover-wrap" style="{style}">'
        f'<table class="fpl-hover-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _cell(value: Any, *, money: bool = False, digits: int | None = None) -> str:
    if money:
        text = _fmt_money(value)
        return html.escape(text) if text else "—"
    if digits is not None:
        n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(n):
            return "—"
        return html.escape(f"{float(n):.{digits}f}")
    text = _plain(value)
    return html.escape(text) if text else "—"


def render_player_frame(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
    *,
    name_keys: set[str] | None = None,
    max_rows: int | None = 80,
    max_height: int | None = None,
) -> None:
    """`columns` is (header, dataframe column). `name_keys` get hover cards."""
    if frame is None or frame.empty:
        return
    work = frame.head(max_rows) if max_rows else frame
    hover_keys = name_keys or {"name"}
    headers = [h for h, _ in columns]
    rows: list[list[str]] = []
    for _, rec in work.iterrows():
        cells: list[str] = []
        for _header, key in columns:
            if key in hover_keys:
                cells.append(name_cell(rec, display=_plain(rec.get(key)) or "—"))
            elif key == "cost_m":
                cells.append(_cell(rec.get(key), money=True))
            elif key in {"xpts", "p_play", "form", "selected_by_percent", "points_per_game"}:
                digits = 2 if key in {"xpts", "p_play"} else 1
                cells.append(_cell(rec.get(key), digits=digits))
            elif key == "points":
                n = pd.to_numeric(pd.Series([rec.get(key)]), errors="coerce").iloc[0]
                cells.append("—" if pd.isna(n) else html.escape(str(int(n))))
            else:
                cells.append(_cell(rec.get(key)))
        rows.append(cells)
    render_html_table(headers, rows, max_height=max_height)
    st.caption("Hover a name for club, value, FPL form, and next 5 (official FDR, 5=hardest).")
