"""Local Streamlit UI. Run via `python -m fpl app`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fpl.config import load_settings
from fpl.ingest.pipeline import run_ingest
from fpl.ingest.history import run_history
from fpl.models.prior import next_unfinished_event
from fpl.optimize.chips import format_chip_report, run_chips
from fpl.optimize.pool import default_season_event
from fpl.optimize.rules import BUDGET_M, SQUAD_SIZE, normalize_position
from fpl.optimize.squad import format_squad_report, run_squad
from fpl.optimize.transfers import (
    fetch_entry_picks,
    format_transfer_report,
    run_transfer,
)
from fpl.ui.advisor import render_advisor
from fpl.ui.help import HOWTO_BODY, HOWTO_TITLE, SIDEBAR_STEPS
from fpl.ui.pitch import fill_from_ids, n_filled, render_squad_pitch, spent_m


def _settings():
    return load_settings()


def _panel_path() -> Path:
    return _settings().processed_dir / "player_gw.parquet"


def _squad_path() -> Path:
    return _settings().overrides_dir / "squad.csv"


def _read_squad_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "element_id" not in frame.columns:
        return []
    ids = pd.to_numeric(frame["element_id"], errors="coerce").dropna().astype(int).tolist()
    return list(dict.fromkeys(ids))


def _write_squad_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"element_id": [int(i) for i in ids]}).to_csv(path, index=False)


def _load_panel() -> pd.DataFrame | None:
    path = _panel_path()
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _pos_or_raw(value):
    if pd.isna(value):
        return value
    try:
        return normalize_position(value)
    except ValueError:
        return value


def _finalize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["element_id"] = pd.to_numeric(out["element_id"], errors="coerce")
    out = out.dropna(subset=["element_id"])
    out["element_id"] = out["element_id"].astype(int)
    out["position"] = out["position"].map(_pos_or_raw)
    out["name"] = out["name"].astype("string").fillna("?")
    out["team"] = out.get("team", pd.Series("?", index=out.index)).astype("string").fillna("?")
    out["cost_m"] = pd.to_numeric(out.get("cost_m"), errors="coerce")
    if "points" not in out.columns:
        out["points"] = pd.to_numeric(out.get("total_points"), errors="coerce").fillna(0)
    else:
        out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0)
    for col in ("form", "selected_by_percent", "points_per_game"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "status" not in out.columns:
        out["status"] = "a"
    out["label"] = (
        out["name"]
        + " · "
        + out["position"].astype("string")
        + " · "
        + out["team"]
        + "  ("
        + out["element_id"].astype(str)
        + ")"
    )
    return out.drop_duplicates("element_id").sort_values(["position", "name"])


def _player_catalog(panel: pd.DataFrame, season: str) -> pd.DataFrame:
    rows = panel.loc[panel["season"].astype(str) == str(season)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["element_id", "name", "position", "team", "team_id", "cost_m", "points"])
    rows["element_id"] = pd.to_numeric(rows["element_id"], errors="coerce")
    rows = rows.dropna(subset=["element_id"])
    rows["element_id"] = rows["element_id"].astype(int)
    points = rows.groupby("element_id")["total_points"].sum() if "total_points" in rows.columns else pd.Series(dtype=float)
    rows = rows.sort_values("event")
    keep = [c for c in ["element_id", "name", "position", "team", "team_id", "cost_m"] if c in rows.columns]
    latest = rows.drop_duplicates("element_id", keep="last")[keep]
    latest["points"] = latest["element_id"].map(points).fillna(0)
    return _finalize_catalog(latest)


def _player_catalog_live(settings) -> pd.DataFrame:
    players_path = settings.processed_dir / "players.parquet"
    if not players_path.exists():
        return pd.DataFrame(columns=["element_id", "name", "position", "team", "team_id", "cost_m", "points"])
    players = pd.read_parquet(players_path)
    teams_path = settings.processed_dir / "teams.parquet"
    teams = pd.read_parquet(teams_path) if teams_path.exists() else pd.DataFrame()
    rows = players.copy()
    if "status" in rows.columns:
        rows = rows[rows["status"].fillna("a").astype(str).str.lower().ne("u")]
    if not teams.empty and "team_id" in teams.columns and "short_name" in teams.columns:
        rows = rows.merge(teams[["team_id", "short_name"]], on="team_id", how="left")
        rows["team"] = rows["short_name"]
    rows["name"] = rows.get("web_name", pd.Series("", index=rows.index)).astype("string").fillna("?")
    rows["points"] = pd.to_numeric(rows.get("total_points"), errors="coerce").fillna(0)
    keep = [
        c
        for c in [
            "element_id",
            "name",
            "position",
            "team",
            "team_id",
            "cost_m",
            "points",
            "form",
            "selected_by_percent",
            "status",
            "points_per_game",
        ]
        if c in rows.columns
    ]
    return _finalize_catalog(rows[keep])


def _xi_table(solution) -> pd.DataFrame:
    table = solution.table.copy()
    table["role"] = "Bench"
    table.loc[table["in_xi"], "role"] = "XI"
    table.loc[table["is_captain"], "role"] = "Captain"
    table.loc[table["is_vice"], "role"] = "Vice"
    cols = [c for c in ["role", "position", "name", "cost_m", "xpts", "p_play"] if c in table.columns]
    return table[cols]


def main() -> None:
    st.set_page_config(page_title="FPL analyser", layout="wide")
    st.title("FPL analyser")
    st.caption("Expected points for your 15 this gameweek. Local engine — nothing is uploaded unless you set OPENAI_API_KEY for the advisor.")

    with st.expander(HOWTO_TITLE, expanded=True):
        st.markdown(HOWTO_BODY)

    cfg = _settings()
    panel = _load_panel()
    live_players = (cfg.processed_dir / "players.parquet").exists()
    live_fixtures = (cfg.processed_dir / "fixtures.parquet").exists()
    if panel is None:
        st.error("No player_gw.parquet yet. Use the sidebar: **Refresh data** → History.")
        seasons: list[str] = []
        default_season, default_event = cfg.current_season, 1
    else:
        seasons = sorted(panel["season"].astype(str).unique().tolist())
        default_season, default_event = default_season_event(panel)

    if live_players and cfg.current_season not in seasons:
        seasons = seasons + [cfg.current_season]
    if live_players:
        default_season = cfg.current_season
        fixtures_path = cfg.processed_dir / "fixtures.parquet"
        events_path = cfg.processed_dir / "events.parquet"
        if live_fixtures:
            try:
                events = pd.read_parquet(events_path) if events_path.exists() else None
                default_event = next_unfinished_event(pd.read_parquet(fixtures_path), events)
            except RuntimeError:
                pass

    if live_players:
        st.info(
            f"Upcoming {cfg.current_season}: xPts uses last completed PL form (mapped by FPL code) "
            "plus this year's prices, fixtures, and official FPL availability. "
            "It is not cup/friendly form and not a news scrape. "
            "**Rebuild 15** is the balanced EV squad."
        )
    elif panel is not None and cfg.current_season not in seasons:
        st.warning(
            "No live FPL snapshot yet. Run **Ingest FPL snapshot** so squad/transfer can score "
            "the upcoming gameweek instead of last season."
        )

    with st.sidebar:
        st.header("This week")
        st.markdown(SIDEBAR_STEPS)
        st.divider()
        st.header("Gameweek")
        season = st.selectbox(
            "Season",
            options=seasons or [default_season],
            index=(seasons.index(default_season) if default_season in seasons else 0),
        )
        event = st.number_input("Gameweek", min_value=1, max_value=38, value=int(default_event), step=1)
        free_transfers = st.number_input("Free transfers", min_value=0, max_value=5, value=1, step=1)
        st.divider()
        st.header("Refresh data")
        st.caption("Ingest is quick. History can take several minutes (one API call per player).")
        if st.button("Ingest FPL snapshot"):
            with st.spinner("Ingesting…"):
                try:
                    run_ingest(refresh=True)
                    st.success("Ingest done.")
                except Exception as exc:
                    st.error(str(exc))
        if st.button("Rebuild history panel"):
            with st.spinner("Building player_gw… this can be slow."):
                try:
                    run_history(refresh=False)
                    st.success("History done. Reload the page.")
                except Exception as exc:
                    st.error(str(exc))

    if str(season) == cfg.current_season and live_players:
        catalog = _player_catalog_live(cfg)
    else:
        catalog = _player_catalog(panel, season) if panel is not None else pd.DataFrame()

    st.subheader("1. Your 15")
    st.caption(
        "Click a shirt to fill that position. Filter the list on the right, **Place on pitch**, "
        "then **Save squad** (2/5/5/3, ≤ £100m, max 3 per club). Or load your FPL team ID below."
    )
    saved_ids = _read_squad_ids(_squad_path())
    valid_ids = set(catalog["element_id"].tolist()) if not catalog.empty else set()
    missing = [i for i in saved_ids if i not in valid_ids]
    if missing:
        st.info(f"Saved IDs not in {season} catalog (kept on disk): {missing}")

    catalog_key = str(season)
    if st.session_state.get("squad_catalog_key") != catalog_key or "squad_slots" not in st.session_state:
        st.session_state.squad_slots = fill_from_ids(saved_ids, catalog)
        st.session_state.squad_catalog_key = catalog_key
        st.session_state.active_slot = None
    else:
        for pos, ids in st.session_state.squad_slots.items():
            st.session_state.squad_slots[pos] = [eid if eid in valid_ids else None for eid in ids]

    if catalog.empty:
        st.warning("No player catalog for this season. Run ingest / history first.")
        picked_ids: list[int] = []
    else:
        picked_ids = render_squad_pitch(catalog)

    col_save, col_fpl, col_id = st.columns([1, 1, 2])
    with col_save:
        can_save = (
            n_filled(st.session_state.get("squad_slots", {})) == SQUAD_SIZE
            and spent_m(st.session_state.get("squad_slots", {}), catalog) <= BUDGET_M + 1e-9
        )
        if st.button("Save squad", type="primary", disabled=not can_save):
            _write_squad_ids(_squad_path(), picked_ids)
            st.success(f"Wrote {_squad_path()}")
    with col_fpl:
        team_id = st.number_input("FPL team ID (optional)", min_value=0, value=0, step=1)
    with col_id:
        if st.button("Load picks from FPL", disabled=int(team_id) <= 0 or catalog.empty):
            try:
                ids, bank = fetch_entry_picks(int(team_id), int(event), settings=_settings())
                st.session_state.squad_slots = fill_from_ids(ids, catalog)
                _write_squad_ids(_squad_path(), [i for i in ids if i in valid_ids][:SQUAD_SIZE])
                st.success(f"Loaded {len(ids)} picks (bank £{bank:.1f}m).")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("2. Run")
    st.caption(
        "**Transfers** needs a saved 15 and answers TAKE vs HOLD. **Chip EV** does not play a chip. "
        "**Rebuild 15** is a wildcard-shaped team from scratch, not an edit of yours."
    )
    c1, c2, c3 = st.columns(3)
    report_box = st.empty()
    table_box = st.empty()

    def _show_text(text: str) -> None:
        report_box.code(text, language=None)

    with c1:
        if st.button("Recommend transfers", help="Hold vs 0–3 moves. Needs a saved 15."):
            if len(_read_squad_ids(_squad_path())) != SQUAD_SIZE:
                st.error("Save 15 players first.")
            else:
                with st.spinner("Solving transfers…"):
                    try:
                        result = run_transfer(
                            season=str(season),
                            event=int(event),
                            squad_path=_squad_path(),
                            free_transfers=int(free_transfers),
                        )
                        text = format_transfer_report(result["plan"])
                        if result.get("note"):
                            text = result["note"] + "\n\n" + text
                        _show_text(text)
                        table_box.dataframe(_xi_table(result["plan"].chosen), hide_index=True)
                    except Exception as exc:
                        st.error(str(exc))
    with c2:
        if st.button("Chip EV", help="This-GW BB / TC / FH / WC. Does not auto-play a chip."):
            if len(_read_squad_ids(_squad_path())) != SQUAD_SIZE:
                st.error("Save 15 players first.")
            else:
                with st.spinner("Scoring chips…"):
                    try:
                        result = run_chips(
                            season=str(season),
                            event=int(event),
                            squad_path=_squad_path(),
                            free_transfers=int(free_transfers),
                        )
                        text = format_chip_report(result["report"])
                        if result.get("note"):
                            text = result["note"] + "\n\n" + text
                        _show_text(text)
                        table_box.dataframe(_xi_table(result["plan"].chosen), hide_index=True)
                    except Exception as exc:
                        st.error(str(exc))
    with c3:
        if st.button("Rebuild 15 from scratch", help="Wildcard-shaped squad. Ignores your current 15."):
            with st.spinner("Solving squad…"):
                try:
                    result = run_squad(season=str(season), event=int(event))
                    text = format_squad_report(result["solution"])
                    if result.get("note"):
                        text = result["note"] + "\n\n" + text
                    _show_text(text)
                    table_box.dataframe(_xi_table(result["solution"]), hide_index=True)
                except Exception as exc:
                    st.error(str(exc))

    st.caption(
        "TAKE/HOLD is expected net vs doing nothing, after 4-point hits. "
        "A starred chip on the chip report is this-GW arithmetic — do not blindly triple a one-week spike."
    )

    brief_ids = picked_ids if len(set(picked_ids)) == SQUAD_SIZE else saved_ids
    render_advisor(
        season=str(season),
        event=int(event),
        free_transfers=int(free_transfers),
        squad_ids=brief_ids,
    )


if __name__ == "__main__":
    main()
