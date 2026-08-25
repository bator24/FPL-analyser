"""FPL-style 2-5-5-3 pitch picker for the local Streamlit app."""

from __future__ import annotations

import unicodedata
from typing import Any

import pandas as pd
import streamlit as st

from fpl.optimize.rules import BUDGET_M, MAX_PER_CLUB, SQUAD_COUNTS, SQUAD_SIZE, normalize_position

PITCH_ROWS = (("FWD", 3), ("MID", 5), ("DEF", 5), ("GKP", 2))
SORT_OPTIONS = (
    "Points (high)",
    "Value (cheap)",
    "Value (dear)",
    "Form",
    "Ownership",
    "Name",
)

_PITCH_CSS = """
<style>
div[data-testid="stVerticalBlock"]:has(> div > span.fpl-pitch-marker) {
  background: #1b7a38;
  background-image:
    linear-gradient(to bottom, rgba(255,255,255,0.06) 0 12%, transparent 12% 88%, rgba(0,0,0,0.12) 88% 100%),
    repeating-linear-gradient(
      to bottom,
      transparent 0,
      transparent 46px,
      rgba(255,255,255,0.07) 46px,
      rgba(255,255,255,0.07) 48px
    );
  border-radius: 16px;
  padding: 0.85rem 0.7rem 1.1rem;
  margin-bottom: 0.6rem;
}
div[data-testid="stVerticalBlock"]:has(> div > span.fpl-pitch-marker) [data-testid="stButton"] > button {
  min-height: 5.1rem;
  width: 100%;
  white-space: pre-wrap;
  font-size: 0.78rem;
  line-height: 1.15;
  border-radius: 10px;
  border: 2px solid #00ff87;
  background: #37003c;
  color: #fff;
}
div[data-testid="stVerticalBlock"]:has(> div > span.fpl-pitch-marker) [class*="st-key-rm_"] [data-testid="stButton"] > button {
  min-height: 1.7rem;
  padding: 0.1rem 0.3rem;
  font-size: 0.72rem;
  background: #111;
  border: 1px solid #fff;
  color: #fff;
}
div[data-testid="stVerticalBlock"]:has(> div > span.fpl-pitch-marker) [data-testid="stButton"] > button:hover {
  border-color: #fff;
  color: #fff;
}
.fpl-pitch-caption {
  color: #e8ffe8;
  text-align: center;
  font-size: 0.85rem;
  margin: 0 0 0.4rem 0;
}
</style>
"""


def empty_slots() -> dict[str, list[int | None]]:
    return {pos: [None] * n for pos, n in SQUAD_COUNTS.items()}


def flatten_slots(slots: dict[str, list[int | None]]) -> list[int]:
    ids: list[int] = []
    for pos, _n in PITCH_ROWS:
        for eid in slots.get(pos, []):
            if eid is not None:
                ids.append(int(eid))
    return ids


def n_filled(slots: dict[str, list[int | None]]) -> int:
    return len(flatten_slots(slots))


def fill_from_ids(ids: list[int], catalog: pd.DataFrame) -> dict[str, list[int | None]]:
    slots = empty_slots()
    if catalog is None or catalog.empty or not ids:
        return slots
    work = catalog.copy()
    work["element_id"] = pd.to_numeric(work["element_id"], errors="coerce")
    work = work.dropna(subset=["element_id"])
    work["element_id"] = work["element_id"].astype(int)
    work["position"] = work["position"].map(_safe_pos)
    lookup = work.drop_duplicates("element_id").set_index("element_id")
    used: set[int] = set()
    for eid in ids:
        eid = int(eid)
        if eid in used or eid not in lookup.index:
            continue
        pos = str(lookup.loc[eid, "position"])
        if pos not in slots:
            continue
        for i, current in enumerate(slots[pos]):
            if current is None:
                slots[pos][i] = eid
                used.add(eid)
                break
    return slots


def _safe_pos(value: Any) -> str:
    try:
        return normalize_position(value)
    except ValueError:
        return str(value).strip().upper()


def _player_row(catalog: pd.DataFrame, element_id: int | None) -> pd.Series | None:
    if element_id is None or catalog.empty:
        return None
    hit = catalog.loc[pd.to_numeric(catalog["element_id"], errors="coerce") == int(element_id)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def spent_m(slots: dict[str, list[int | None]], catalog: pd.DataFrame) -> float:
    total = 0.0
    for eid in flatten_slots(slots):
        row = _player_row(catalog, eid)
        if row is None:
            continue
        cost = _num(row.get("cost_m"))
        if pd.notna(cost):
            total += cost
    return round(total, 1)


def ids_equal(left: list[int], right: list[int]) -> bool:
    return sorted(int(x) for x in left) == sorted(int(x) for x in right)


def save_blockers(slots: dict[str, list[int | None]], catalog: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    filled = n_filled(slots)
    if filled != SQUAD_SIZE:
        reasons.append(f"{filled}/{SQUAD_SIZE} selected")
    spent = spent_m(slots, catalog)
    if spent > BUDGET_M + 1e-9:
        reasons.append(f"£{spent:.1f}m / £{BUDGET_M:.1f}m")
    over = club_counts(slots, catalog)
    over = over[over > MAX_PER_CLUB]
    if not over.empty:
        reasons.append(f"more than {MAX_PER_CLUB} from one club")
    return reasons


def club_counts(slots: dict[str, list[int | None]], catalog: pd.DataFrame) -> pd.Series:
    names: list[str] = []
    for eid in flatten_slots(slots):
        row = _player_row(catalog, eid)
        if row is None:
            continue
        club = row.get("team_id")
        if pd.isna(club):
            club = row.get("team")
        names.append(str(club))
    if not names:
        return pd.Series(dtype=int)
    return pd.Series(names).value_counts()


def would_break_club_cap(
    slots: dict[str, list[int | None]],
    catalog: pd.DataFrame,
    element_id: int,
    *,
    replacing: int | None = None,
) -> bool:
    tentative = [eid for eid in flatten_slots(slots) if eid != replacing]
    tentative.append(int(element_id))
    names: list[str] = []
    for eid in tentative:
        row = _player_row(catalog, eid)
        if row is None:
            continue
        club = row.get("team_id")
        if pd.isna(club):
            club = row.get("team")
        names.append(str(club))
    if not names:
        return False
    return bool(pd.Series(names).value_counts().max() > MAX_PER_CLUB)


def fold_search(text: Any) -> str:
    """Casefold and strip accents so 'Salah' / 'fernandes' / 'kadioglu' match."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ı", "i").replace("İ", "i").casefold()


def search_blob(frame: pd.DataFrame) -> pd.Series:
    parts = [frame["name"].astype("string").fillna("")]
    for column in ("first_name", "second_name", "web_name", "team"):
        if column in frame.columns:
            parts.append(frame[column].astype("string").fillna(""))
    joined = parts[0]
    for extra in parts[1:]:
        joined = joined + " " + extra
    return joined.map(fold_search)


def filter_candidates(
    catalog: pd.DataFrame,
    *,
    position: str | None,
    exclude_ids: set[int],
    name: str = "",
    min_cost: float | None = None,
    max_cost: float | None = None,
    min_points: float | None = None,
    teams: list[str] | None = None,
    sort_by: str = SORT_OPTIONS[0],
) -> pd.DataFrame:
    if catalog is None or catalog.empty:
        return pd.DataFrame()
    out = catalog.copy()
    out["position"] = out["position"].map(_safe_pos)
    out["element_id"] = pd.to_numeric(out["element_id"], errors="coerce")
    out = out.dropna(subset=["element_id"])
    out["element_id"] = out["element_id"].astype(int)
    if position:
        out = out.loc[out["position"] == position]
    out = out.loc[~out["element_id"].isin(exclude_ids)]
    if name.strip():
        needle = fold_search(name.strip())
        out = out.loc[search_blob(out).str.contains(needle, regex=False, na=False)]
    cost = pd.to_numeric(out.get("cost_m"), errors="coerce")
    if min_cost is not None:
        out = out.loc[cost >= float(min_cost)]
        cost = pd.to_numeric(out.get("cost_m"), errors="coerce")
    if max_cost is not None:
        out = out.loc[cost <= float(max_cost)]
    points = pd.to_numeric(out.get("points"), errors="coerce").fillna(0)
    if min_points is not None:
        out = out.loc[points >= float(min_points)]
    if teams:
        out = out.loc[out["team"].astype("string").isin(teams)]
    return _sort_candidates(out, sort_by)


def _sort_candidates(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if sort_by == "Value (cheap)":
        return out.sort_values(["cost_m", "name"], ascending=[True, True], na_position="last")
    if sort_by == "Value (dear)":
        return out.sort_values(["cost_m", "name"], ascending=[False, True], na_position="last")
    if sort_by == "Form":
        return out.sort_values(["form", "points", "name"], ascending=[False, False, True], na_position="last")
    if sort_by == "Ownership":
        return out.sort_values(
            ["selected_by_percent", "points", "name"],
            ascending=[False, False, True],
            na_position="last",
        )
    if sort_by == "Name":
        return out.sort_values("name", ascending=True, na_position="last")
    return out.sort_values(["points", "cost_m", "name"], ascending=[False, True, True], na_position="last")


def _row_layout(n: int) -> list[int | None]:
    if n == 2:
        return [None, 0, None, 1, None]
    if n == 3:
        return [None, 0, 1, 2, None]
    return list(range(n))


def _slot_label(row: pd.Series | None, position: str) -> str:
    if row is None:
        return f"+ {position}"
    name = str(row.get("name") or "?")
    team = str(row.get("team") or "")
    cost = pd.to_numeric(pd.Series([row.get("cost_m")]), errors="coerce").iloc[0]
    pts = pd.to_numeric(pd.Series([row.get("points")]), errors="coerce").iloc[0]
    cost_s = f"£{cost:.1f}" if pd.notna(cost) else ""
    pts_s = f"{int(pts)} pts" if pd.notna(pts) else ""
    return f"{name}\n{team} {cost_s}\n{pts_s}".strip()


def _inject_css() -> None:
    if st.session_state.get("_fpl_pitch_css"):
        return
    st.markdown(_PITCH_CSS, unsafe_allow_html=True)
    st.session_state["_fpl_pitch_css"] = True


def render_squad_pitch(catalog: pd.DataFrame) -> list[int]:
    """Pitch + position-filtered picker. Returns currently selected element_ids (may be <15)."""
    _inject_css()
    if "squad_slots" not in st.session_state:
        st.session_state.squad_slots = empty_slots()
    if "active_slot" not in st.session_state:
        st.session_state.active_slot = None

    slots: dict[str, list[int | None]] = st.session_state.squad_slots
    filled = n_filled(slots)
    spent = spent_m(slots, catalog)
    bank = round(BUDGET_M - spent, 1)
    counts = {pos: sum(x is not None for x in slots[pos]) for pos in SQUAD_COUNTS}
    count_txt = " · ".join(f"{counts[p]}/{n} {p}" for p, n in SQUAD_COUNTS.items())

    pitch, picker = st.columns([1.35, 1], gap="large")
    with pitch:
        with st.container():
            st.markdown('<span class="fpl-pitch-marker"></span>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="fpl-pitch-caption">{filled}/{SQUAD_SIZE} · '
                f"£{spent:.1f}m spent · £{bank:.1f}m ITB · {count_txt}</p>",
                unsafe_allow_html=True,
            )
            for pos, n in PITCH_ROWS:
                layout = _row_layout(n)
                cols = st.columns(len(layout), gap="small")
                for col, slot_i in zip(cols, layout):
                    with col:
                        if slot_i is None:
                            st.write("")
                            continue
                        row = _player_row(catalog, slots[pos][slot_i])
                        active = st.session_state.active_slot == (pos, slot_i)
                        if st.button(
                            _slot_label(row, pos),
                            key=f"pitch_{pos}_{slot_i}",
                            use_container_width=True,
                            type="primary" if active else "secondary",
                        ):
                            st.session_state.active_slot = (pos, slot_i)
                            st.rerun()
                        if slots[pos][slot_i] is not None:
                            if st.button(
                                "Remove",
                                key=f"rm_{pos}_{slot_i}",
                                use_container_width=True,
                            ):
                                slots[pos][slot_i] = None
                                st.session_state.squad_slots = slots
                                st.session_state.active_slot = (pos, slot_i)
                                st.rerun()

    with picker:
        _render_picker(catalog, slots)

    over = club_counts(slots, catalog)
    over = over[over > MAX_PER_CLUB]
    if not over.empty:
        st.warning(f"More than {MAX_PER_CLUB} from one club: {over.to_dict()}")
    if spent > BUDGET_M + 1e-9:
        st.warning(f"Over budget: £{spent:.1f}m / £{BUDGET_M:.1f}m")
    return flatten_slots(slots)


def _first_empty_slot(slots: dict[str, list[int | None]], position: str) -> tuple[str, int] | None:
    for i, eid in enumerate(slots.get(position, [])):
        if eid is None:
            return (position, i)
    return None


def _player_choice_label(row: pd.Series) -> str:
    pts = pd.to_numeric(row.get("points"), errors="coerce")
    cost = pd.to_numeric(row.get("cost_m"), errors="coerce")
    pts_s = 0 if pd.isna(pts) else int(pts)
    cost_s = 0.0 if pd.isna(cost) else float(cost)
    pos = str(row.get("position") or "")
    return f"{row['name']} · {pos} · {row['team']} · £{cost_s:.1f} · {pts_s} pts"


def _render_picker(catalog: pd.DataFrame, slots: dict[str, list[int | None]]) -> None:
    active = st.session_state.active_slot
    st.markdown("**Pick a player**")
    st.caption("Type a name (Salah, Fernandes, …). You can search before clicking a shirt.")

    pos: str | None
    slot_i: int | None
    current: int | None
    if active is None:
        pos, slot_i, current = None, None, None
    else:
        pos, slot_i = active
        current = slots[pos][slot_i]
        need = SQUAD_COUNTS[pos]
        have = sum(x is not None for x in slots[pos])
        st.caption(f"Selected shirt: {pos} slot {slot_i + 1} · {have}/{need} filled")
        if current is not None:
            who = _player_row(catalog, current)
            label = str(who["name"]) if who is not None else "this player"
            if st.button(f"Remove {label}", key="picker_remove", use_container_width=True):
                slots[pos][slot_i] = None
                st.session_state.squad_slots = slots
                st.session_state.active_slot = (pos, slot_i)
                st.rerun()

    occupied = set(flatten_slots(slots))
    if current is not None:
        occupied.discard(int(current))

    query = st.text_input("Search", key="pick_name", placeholder="Name or club…", label_visibility="visible")
    teams = sorted(catalog["team"].dropna().astype(str).unique().tolist()) if not catalog.empty else []
    team_sel = st.multiselect("Club", options=teams, key="pick_teams")
    costs = pd.to_numeric(catalog.get("cost_m"), errors="coerce") if not catalog.empty else pd.Series(dtype=float)
    cmin = float(costs.min()) if costs.notna().any() else 4.0
    cmax = float(costs.max()) if costs.notna().any() else 15.0
    price = st.slider(
        "Price (£m)",
        min_value=round(cmin, 1),
        max_value=round(cmax, 1),
        value=(round(cmin, 1), round(cmax, 1)),
        step=0.5,
        key="pick_price",
    )
    min_pts = st.number_input("Min total points", min_value=0, value=0, step=1, key="pick_min_pts")
    sort_by = st.selectbox("Sort", options=list(SORT_OPTIONS), key="pick_sort")

    # Empty search + a selected shirt → that position. A name query searches everyone.
    position_filter = None if str(query).strip() else pos
    if position_filter is None and not str(query).strip() and active is None:
        st.info("Click a shirt on the pitch, or type a name above.")
        return

    filtered = filter_candidates(
        catalog,
        position=position_filter,
        exclude_ids=occupied,
        name=str(query or ""),
        min_cost=price[0],
        max_cost=price[1],
        min_points=float(min_pts),
        teams=team_sel,
        sort_by=str(sort_by),
    )
    show_cols = [
        c
        for c in ["name", "position", "team", "cost_m", "points", "form", "selected_by_percent", "status"]
        if c in filtered.columns
    ]
    rename = {
        "name": "Player",
        "position": "Pos",
        "team": "Club",
        "cost_m": "£m",
        "points": "Pts",
        "form": "Form",
        "selected_by_percent": "Own %",
        "status": "Status",
    }
    if filtered.empty:
        st.info("No players match that search.")
        if current is not None and st.button("Clear slot", use_container_width=True):
            slots[pos][slot_i] = None
            st.session_state.squad_slots = slots
            st.rerun()
        return

    st.dataframe(
        filtered[show_cols].rename(columns=rename).head(80),
        hide_index=True,
        use_container_width=True,
        height=280,
    )
    labels = [_player_choice_label(row) for _, row in filtered.iterrows()]
    id_by_label = dict(zip(labels, filtered["element_id"].astype(int).tolist()))
    # Duplicate labels are rare (same display); keep first.
    unique_labels = list(dict.fromkeys(labels))
    choice_key = "pick_choice_label"
    if st.session_state.get(choice_key) not in unique_labels:
        st.session_state.pop(choice_key, None)

    chosen_label = st.selectbox(
        "Choose (type here too)",
        options=unique_labels,
        key=choice_key,
    )
    chosen_id = int(id_by_label[chosen_label])
    chosen_pos = str(filtered.loc[filtered["element_id"] == chosen_id, "position"].iloc[0])
    chosen_pos = _safe_pos(chosen_pos)

    def _target_slot() -> tuple[str, int] | None:
        if pos is not None and slot_i is not None and chosen_pos == pos:
            return (pos, slot_i)
        return _first_empty_slot(slots, chosen_pos)

    place, clear = st.columns(2)
    with place:
        if st.button("Place on pitch", type="primary", use_container_width=True):
            target = _target_slot()
            if target is None:
                st.error(f"No empty {chosen_pos} shirt. Clear one first.")
            elif would_break_club_cap(
                slots, catalog, chosen_id, replacing=slots[target[0]][target[1]]
            ):
                st.error(f"Max {MAX_PER_CLUB} players per club.")
            else:
                tpos, ti = target
                slots[tpos][ti] = chosen_id
                st.session_state.squad_slots = slots
                st.session_state.active_slot = _next_empty(slots, tpos, ti)
                st.rerun()
    with clear:
        if st.button("Clear slot", use_container_width=True, disabled=current is None):
            slots[pos][slot_i] = None
            st.session_state.squad_slots = slots
            st.rerun()


def _next_empty(
    slots: dict[str, list[int | None]],
    pos: str,
    slot_i: int,
) -> tuple[str, int] | None:
    for j, eid in enumerate(slots[pos]):
        if eid is None and j != slot_i:
            return (pos, j)
    for other, n in PITCH_ROWS:
        for j, eid in enumerate(slots[other]):
            if eid is None:
                return (other, j)
    return (pos, slot_i)
