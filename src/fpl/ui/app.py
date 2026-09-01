"""Local Streamlit UI. Run via `python -m fpl app`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fpl.config import load_settings
from fpl.ingest.pipeline import run_ingest
from fpl.ingest.history import run_history
from fpl.models.horizon import attach_horizon
from fpl.models.prior import next_unfinished_event
from fpl.optimize.chips import run_chips
from fpl.optimize.pool import default_season_event
from fpl.optimize.rules import SQUAD_SIZE, normalize_position
from fpl.optimize.squad import run_squad
from fpl.optimize.transfers import fetch_entry_picks, run_transfer
from fpl.ui.advisor import render_advisor
from fpl.ui.chrome import inject_chrome, render_topbar, status_pill
from fpl.ui.help import HOWTO_BODY, HOWTO_TITLE, SIDEBAR_STEPS
from fpl.ui.panels import render_chip_report, render_squad_solution, render_transfer_plan
from fpl.ui.pitch import (
    fill_from_ids,
    flatten_slots,
    ids_equal,
    render_squad_pitch,
    save_blockers,
)


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
            "first_name",
            "second_name",
            "web_name",
            "position",
            "team",
            "team_id",
            "cost_m",
            "points",
            "form",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "selected_by_percent",
            "status",
            "news",
            "chance_of_playing_next_round",
            "points_per_game",
            "event_points",
            "transfers_in_event",
            "transfers_out_event",
        ]
        if c in rows.columns
    ]
    return _finalize_catalog(rows[keep])


def _with_horizon(catalog: pd.DataFrame, settings, event: int) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    fixtures_path = settings.processed_dir / "fixtures.parquet"
    teams_path = settings.processed_dir / "teams.parquet"
    if not fixtures_path.exists():
        return catalog
    fixtures = pd.read_parquet(fixtures_path)
    teams = pd.read_parquet(teams_path) if teams_path.exists() else pd.DataFrame()
    return attach_horizon(catalog, fixtures, teams, from_event=int(event))


def _need_saved_15() -> bool:
    return len(_read_squad_ids(_squad_path())) != SQUAD_SIZE


def main() -> None:
    st.set_page_config(page_title="FPL analyser", layout="wide", initial_sidebar_state="expanded")
    inject_chrome()

    cfg = _settings()
    panel = _load_panel()
    live_players = (cfg.processed_dir / "players.parquet").exists()
    live_fixtures = (cfg.processed_dir / "fixtures.parquet").exists()
    if panel is None:
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
    catalog = _with_horizon(catalog, cfg, int(event))

    saved_ids = _read_squad_ids(_squad_path())
    valid_ids = set(catalog["element_id"].tolist()) if not catalog.empty else set()
    catalog_key = str(season)
    if st.session_state.get("squad_catalog_key") != catalog_key or "squad_slots" not in st.session_state:
        st.session_state.squad_slots = fill_from_ids(saved_ids, catalog)
        st.session_state.squad_catalog_key = catalog_key
        st.session_state.active_slot = None
    else:
        for pos, ids in st.session_state.squad_slots.items():
            st.session_state.squad_slots[pos] = [eid if eid in valid_ids else None for eid in ids]

    picked_ids = flatten_slots(st.session_state.squad_slots)
    dirty = not ids_equal(picked_ids, saved_ids)
    if len(saved_ids) == SQUAD_SIZE and not dirty:
        pill = status_pill("ok", "saved on disk")
    elif dirty:
        pill = status_pill("warn", "unsaved changes")
    else:
        pill = status_pill("mute", "no squad saved")
    render_topbar(season=str(season), event=int(event), status_html=pill)
    st.caption(
        "Local expected points for **your 15 this gameweek**. "
        "Save writes `data/overrides/squad.csv` on this machine — not FPL, not the cloud. "
        "Nothing is uploaded unless you set OPENAI_API_KEY for the advisor."
    )

    with st.expander(HOWTO_TITLE, expanded=False):
        st.markdown(HOWTO_BODY)

    if panel is None:
        st.error("No player_gw.parquet yet. Sidebar → **Rebuild history panel**.")
    if live_players:
        st.caption(
            f"Upcoming {cfg.current_season}: xPts uses last completed PL form (mapped by FPL code) "
            "plus this year's prices, fixtures, and official FPL availability — not cup/friendly form, not a news scrape."
        )
    elif panel is not None and cfg.current_season not in seasons:
        st.warning(
            "No live FPL snapshot yet. Run **Ingest FPL snapshot** so squad/transfer can score "
            "the upcoming gameweek instead of last season."
        )

    team_tab, xfer_tab, chip_tab, wild_tab, advisor_tab = st.tabs(
        ["My Team", "Transfers", "Chips", "Wildcard 15", "Advisor"]
    )

    with team_tab:
        _render_my_team(
            catalog=catalog,
            season=str(season),
            event=int(event),
            saved_ids=saved_ids,
            valid_ids=valid_ids,
            dirty=dirty,
        )

    with xfer_tab:
        _render_transfers_tab(
            season=str(season),
            event=int(event),
            free_transfers=int(free_transfers),
            catalog=catalog,
        )

    with chip_tab:
        _render_chips_tab(season=str(season), event=int(event), free_transfers=int(free_transfers))

    with wild_tab:
        _render_wildcard_tab(season=str(season), event=int(event), catalog=catalog)

    with advisor_tab:
        # Prefer the pitch if it is a full 15; otherwise the last file on disk.
        live_ids = flatten_slots(st.session_state.get("squad_slots", {}))
        brief_ids = live_ids if len(set(live_ids)) == SQUAD_SIZE else saved_ids
        render_advisor(
            season=str(season),
            event=int(event),
            free_transfers=int(free_transfers),
            squad_ids=brief_ids,
        )


def _render_my_team(
    *,
    catalog: pd.DataFrame,
    season: str,
    event: int,
    saved_ids: list[int],
    valid_ids: set[int],
    dirty: bool,
) -> None:
    missing = [i for i in saved_ids if i not in valid_ids]
    if missing:
        st.info(f"Saved IDs not in {season} catalog (kept on disk): {missing}")

    if catalog.empty:
        st.warning("No player catalog for this season. Run ingest / history first.")
        return

    picked_ids = render_squad_pitch(catalog)
    blockers = save_blockers(st.session_state.get("squad_slots", {}), catalog)
    can_save = not blockers

    left, right = st.columns([1.2, 1.1])
    with left:
        if len(saved_ids) == SQUAD_SIZE and not dirty:
            st.success(f"This 15 is saved at `{_squad_path()}`. Restarting the app keeps it.")
        elif dirty:
            st.warning("Pitch differs from the file on disk. **Save squad** or those shirts are only in this browser session.")
        else:
            st.info(
                f"Save writes `{_squad_path()}` (15 `element_id`s). "
                "That file is what Transfers / Chips / the CLI read. Closing the tab does not upload anything to FPL."
            )
        if blockers:
            st.caption("Save unlocks at 2/5/5/3, ≤ £100.0m, max 3 per club. Now: " + "; ".join(blockers))
        if st.button("Save squad", type="primary", disabled=not can_save):
            _write_squad_ids(_squad_path(), picked_ids)
            st.rerun()
    with right:
        st.markdown("**Load from FPL**")
        st.caption("Public entry ID from `fantasy.premierleague.com/entry/…` — not a login.")
        team_id = st.number_input("FPL team ID", min_value=0, value=0, step=1)
        if st.button("Load picks from FPL", disabled=int(team_id) <= 0):
            try:
                ids, bank = fetch_entry_picks(int(team_id), int(event), settings=_settings())
                st.session_state.squad_slots = fill_from_ids(ids, catalog)
                _write_squad_ids(_squad_path(), [i for i in ids if i in valid_ids][:SQUAD_SIZE])
                st.success(f"Loaded {len(ids)} picks (bank £{bank:.1f}m) and saved to disk.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    last = st.session_state.get("last_transfer_plan")
    if last is not None:
        st.divider()
        st.markdown("**Last transfer score** (Transfers tab)")
        render_transfer_plan(last, note=st.session_state.get("last_transfer_note"), catalog=catalog)


def _render_transfers_tab(
    *,
    season: str,
    event: int,
    free_transfers: int,
    catalog: pd.DataFrame,
) -> None:
    st.caption(
        "TAKE vs HOLD for the **saved** 15, after 4-point hits. "
        "This is expected points, not mini-league rank and not a 0–100 team rating."
    )
    if st.button("Score transfers", type="primary"):
        if _need_saved_15():
            st.error("Save a legal 15 on My Team first.")
        else:
            with st.spinner("Solving transfers…"):
                try:
                    result = run_transfer(
                        season=str(season),
                        event=int(event),
                        squad_path=_squad_path(),
                        free_transfers=int(free_transfers),
                    )
                    st.session_state.last_transfer_plan = result["plan"]
                    st.session_state.last_transfer_note = result.get("note")
                except Exception as exc:
                    st.error(str(exc))
                    return
    plan = st.session_state.get("last_transfer_plan")
    if plan is None:
        st.info("Save your 15, then score. Recommendation uses the file on disk, not unsaved shirts.")
        return
    render_transfer_plan(plan, note=st.session_state.get("last_transfer_note"), catalog=catalog)


def _render_chips_tab(*, season: str, event: int, free_transfers: int) -> None:
    st.caption("This-GW EV only. Does **not** play a chip. A starred line is arithmetic, not an instruction.")
    if st.button("Score chips", type="primary"):
        if _need_saved_15():
            st.error("Save a legal 15 on My Team first.")
        else:
            with st.spinner("Scoring chips…"):
                try:
                    result = run_chips(
                        season=str(season),
                        event=int(event),
                        squad_path=_squad_path(),
                        free_transfers=int(free_transfers),
                    )
                    st.session_state.last_chip_report = result["report"]
                    st.session_state.last_chip_note = result.get("note")
                except Exception as exc:
                    st.error(str(exc))
                    return
    report = st.session_state.get("last_chip_report")
    if report is None:
        st.info("Save your 15 on My Team, then score chips here.")
        return
    render_chip_report(report, note=st.session_state.get("last_chip_note"))


def _render_wildcard_tab(*, season: str, event: int, catalog: pd.DataFrame) -> None:
    st.caption("Rebuilds a legal 15 from scratch. Ignores your current team — wildcard-shaped comparison, not an edit.")
    if st.button("Rebuild 15 from scratch", type="primary"):
        with st.spinner("Solving squad…"):
            try:
                result = run_squad(season=str(season), event=int(event))
                st.session_state.last_squad_solution = result["solution"]
                st.session_state.last_squad_note = result.get("note")
            except Exception as exc:
                st.error(str(exc))
                return
    solution = st.session_state.get("last_squad_solution")
    if solution is None:
        st.info("Run this when you want a from-scratch EV 15 to compare against yours.")
        return
    render_squad_solution(solution, note=st.session_state.get("last_squad_note"), catalog=catalog)


if __name__ == "__main__":
    main()
