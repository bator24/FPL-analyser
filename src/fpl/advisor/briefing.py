"""Weekly manager briefing: last completed GW recap + this-GW engine recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fpl.advisor.chat import take_argument
from fpl.config import Settings, load_settings
from fpl.models.horizon import enrich_rows, load_player_context, merge_context
from fpl.models.prior import ensure_code_map, panel_has_gameweek, prior_season_key
from fpl.optimize.chips import format_chip_report, run_chips
from fpl.optimize.rules import SQUAD_SIZE, normalize_position
from fpl.optimize.transfers import format_transfer_report


def last_finished_event(fixtures: pd.DataFrame) -> int | None:
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return None
    fx = fixtures.copy()
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    finished = fx["finished"].fillna(False).astype(bool) if "finished" in fx.columns else False
    done = fx.loc[finished & fx["event"].notna()]
    if done.empty:
        return None
    return int(done["event"].max())


def _collapse_recap(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    work = rows.copy()
    work["element_id"] = pd.to_numeric(work["element_id"], errors="coerce")
    work = work.dropna(subset=["element_id"])
    work["element_id"] = work["element_id"].astype(int)
    agg: dict[str, Any] = {}
    for col, how in (
        ("name", "first"),
        ("position", "first"),
        ("team", "first"),
        ("minutes", "sum"),
        ("total_points", "sum"),
        ("goals_scored", "sum"),
        ("assists", "sum"),
        ("was_home", "first"),
        ("opponent_short", "first"),
    ):
        if col in work.columns:
            agg[col] = how
    grouped = work.groupby("element_id", sort=False).agg(agg).reset_index()
    if "position" in grouped.columns:
        grouped["position"] = grouped["position"].map(
            lambda v: normalize_position(v) if pd.notna(v) else v
        )
    return grouped


def recap_from_panel(
    panel: pd.DataFrame,
    *,
    season: str,
    event: int,
    squad_ids: list[int],
) -> dict[str, Any]:
    ids = [int(i) for i in squad_ids]
    mask = (panel["season"].astype(str) == str(season)) & (
        pd.to_numeric(panel["event"], errors="coerce") == int(event)
    )
    rows = panel.loc[mask].copy()
    rows["element_id"] = pd.to_numeric(rows["element_id"], errors="coerce")
    rows = rows.loc[rows["element_id"].isin(ids)]
    table = _collapse_recap(rows)
    found = set(pd.to_numeric(table["element_id"], errors="coerce").dropna().astype(int)) if not table.empty else set()
    missing = [i for i in ids if i not in found]
    players: list[dict[str, Any]] = []
    for _, row in table.iterrows() if not table.empty else []:
        mins = float(pd.to_numeric(row.get("minutes"), errors="coerce") or 0)
        pts = float(pd.to_numeric(row.get("total_points"), errors="coerce") or 0)
        players.append(
            {
                "element_id": int(row["element_id"]),
                "name": str(row.get("name") or row["element_id"]),
                "position": str(row.get("position") or "?"),
                "team": str(row.get("team") or ""),
                "minutes": mins,
                "points": pts,
                "blank": pts <= 0,
                "did_not_play": mins <= 0,
            }
        )
    players.sort(key=lambda p: (-p["points"], p["name"]))
    total = float(sum(p["points"] for p in players))
    blanks = [p["name"] for p in players if p["blank"]]
    dnp = [p["name"] for p in players if p["did_not_play"]]
    best = players[0] if players else None
    worst = min(players, key=lambda p: p["points"]) if players else None
    return {
        "season": str(season),
        "event": int(event),
        "source": "player_gw",
        "players": players,
        "n_found": len(players),
        "missing_ids": missing,
        "total_points": total,
        "blanks": blanks,
        "did_not_play": dnp,
        "best": best,
        "worst": worst,
        "note": (
            "Raw combined points for whoever we could match — not FPL XI score "
            "(no captain multiplier, no auto-subs)."
        ),
    }


def recap_via_code(
    panel: pd.DataFrame,
    players: pd.DataFrame,
    *,
    settings: Settings,
    squad_ids: list[int],
    prior_season: str,
    prior_event: int,
) -> dict[str, Any]:
    """Map this year's 15 onto last season's finale using stable FPL `code`."""
    live = players.copy()
    live["element_id"] = pd.to_numeric(live["element_id"], errors="coerce")
    live = live.dropna(subset=["element_id"])
    live["element_id"] = live["element_id"].astype(int)
    if "code" not in live.columns:
        raise RuntimeError("players.parquet has no FPL code column; re-run ingest")
    live["code"] = pd.to_numeric(live["code"], errors="coerce")
    wanted = live.loc[live["element_id"].isin([int(i) for i in squad_ids])]
    code_map = ensure_code_map(settings, prior_season)
    mapped = wanted.merge(code_map, on="code", how="left")
    prior_ids = pd.to_numeric(mapped["prior_element_id"], errors="coerce").dropna().astype(int).tolist()
    recap = recap_from_panel(panel, season=prior_season, event=prior_event, squad_ids=prior_ids)
    recap["source"] = "code_map"
    recap["mapped_from"] = "current element_id via FPL code"
    names: dict[int, str] = {}
    for _, row in mapped.iterrows():
        if pd.notna(row.get("prior_element_id")):
            names[int(row["prior_element_id"])] = str(row.get("web_name") or row.get("name") or row["element_id"])
    for player in recap["players"]:
        player["name"] = names.get(player["element_id"], player["name"])
    recap["missing_ids"] = [
        int(i)
        for i in squad_ids
        if int(i) not in set(pd.to_numeric(mapped.loc[mapped["prior_element_id"].notna(), "element_id"], errors="coerce").dropna().astype(int))
    ]
    recap["n_found"] = len(recap["players"])
    return recap


def build_last_gw_recap(
    *,
    settings: Settings,
    panel: pd.DataFrame,
    squad_ids: list[int],
    current_season: str,
    upcoming_event: int,
) -> dict[str, Any]:
    fixtures_path = settings.processed_dir / "fixtures.parquet"
    players_path = settings.processed_dir / "players.parquet"
    fixtures = pd.read_parquet(fixtures_path) if fixtures_path.exists() else pd.DataFrame()
    last_fin = last_finished_event(fixtures)
    if last_fin is not None and panel_has_gameweek(panel, current_season, last_fin):
        recap = recap_from_panel(panel, season=current_season, event=last_fin, squad_ids=squad_ids)
        recap["headline"] = f"Previous GW: {current_season} GW{last_fin}"
        return recap
    if last_fin is not None and last_fin >= 1:
        return {
            "season": current_season,
            "event": last_fin,
            "source": "missing_panel",
            "players": [],
            "n_found": 0,
            "missing_ids": list(squad_ids),
            "total_points": 0.0,
            "blanks": [],
            "did_not_play": [],
            "best": None,
            "worst": None,
            "headline": f"Previous GW: {current_season} GW{last_fin} (not in history yet)",
            "note": "Run `python -m fpl history` after FPL writes element-summary rows.",
        }
    # Season not started (or no finished fixtures): last season finale for this 15.
    prior = prior_season_key(settings, panel, current_season)
    prior_events = pd.to_numeric(
        panel.loc[panel["season"].astype(str) == prior, "event"], errors="coerce"
    )
    if prior_events.empty or not players_path.exists():
        return {
            "season": prior,
            "event": None,
            "source": "none",
            "players": [],
            "n_found": 0,
            "missing_ids": list(squad_ids),
            "total_points": 0.0,
            "blanks": [],
            "did_not_play": [],
            "best": None,
            "worst": None,
            "headline": "No previous GW to recap",
            "note": "Need a history panel and a live player snapshot.",
        }
    prior_event = int(prior_events.max())
    try:
        recap = recap_via_code(
            panel,
            pd.read_parquet(players_path),
            settings=settings,
            squad_ids=squad_ids,
            prior_season=prior,
            prior_event=prior_event,
        )
    except RuntimeError as exc:
        return {
            "season": prior,
            "event": prior_event,
            "source": "code_map_failed",
            "players": [],
            "n_found": 0,
            "missing_ids": list(squad_ids),
            "total_points": 0.0,
            "blanks": [],
            "did_not_play": [],
            "best": None,
            "worst": None,
            "headline": f"Could not map this 15 onto {prior} GW{prior_event}",
            "note": str(exc),
        }
    recap["headline"] = (
        f"No {current_season} matches in the panel yet. "
        f"Last completed PL for this 15: {prior} GW{prior_event}"
    )
    recap["upcoming_event"] = int(upcoming_event)
    return recap


def _captain_row(table: pd.DataFrame) -> dict[str, Any]:
    row = table.loc[table["is_captain"]].iloc[0]
    return {
        "name": str(row.get("name")),
        "xpts": float(row["xpts"]),
        "p_play": float(row["p_play"]),
        "element_id": int(row["element_id"]),
    }


def upcoming_facts_from_chips(chip_result: dict[str, Any]) -> dict[str, Any]:
    plan = chip_result["plan"]
    report = chip_result["report"]
    hold = plan.hold
    chosen = plan.chosen
    cap = _captain_row(hold.table)
    cap_after = _captain_row(chosen.table)
    xi = hold.table.loc[hold.table["in_xi"]].copy()
    xi_rows = []
    for _, row in xi.iterrows():
        cost = row.get("cost_m")
        xi_rows.append(
            {
                "element_id": int(row["element_id"]),
                "name": str(row.get("name")),
                "position": str(row.get("position")),
                "xpts": float(row["xpts"]),
                "p_play": float(row["p_play"]),
                "captain": bool(row.get("is_captain")),
                "cost_m": None if pd.isna(cost) else float(cost),
            }
        )
    xi_rows.sort(key=lambda r: -r["xpts"])
    action = "HOLD"
    if plan.recommend and plan.n_transfers > 0:
        action = "TAKE TRANSFERS"
    chip_best = report.option(report.best_this_gw)
    return {
        "season": chosen.season,
        "event": chosen.event,
        "action": action,
        "recommend_transfers": bool(plan.recommend),
        "n_transfers": plan.n_transfers,
        "hits": plan.hits,
        "expected_net": float(plan.expected_net),
        "hold_ev": float(hold.objective),
        "chosen_ev": float(chosen.net_objective),
        "transfers_out": plan.transfers_out,
        "transfers_in": plan.transfers_in,
        "alternatives": list(getattr(plan, "alternatives", None) or []),
        "captain_hold": cap,
        "captain_after": cap_after,
        "xi": xi_rows,
        "best_this_gw": report.best_this_gw,
        "best_no_chip": report.best_no_chip,
        "chip_beats_no_chip": report.chip_beats_no_chip,
        "chip_best_label": chip_best.label,
        "chip_note": (
            "Chip table is this-GW arithmetic only. Do not auto-play TC/BB/FH/WC "
            "because a one-week spike looks good."
        ),
        "transfer_report": format_transfer_report(plan),
        "chip_report": format_chip_report(report),
        "engine_note": chip_result.get("note") or "",
    }


def _eid(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    try:
        return int(row["element_id"])
    except (KeyError, TypeError, ValueError):
        return None


def enrich_upcoming(upcoming: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out = dict(upcoming)
    out["transfers_out"] = enrich_rows(list(out.get("transfers_out") or []), by_id)
    out["transfers_in"] = enrich_rows(list(out.get("transfers_in") or []), by_id)
    alts = []
    for alt in out.get("alternatives") or []:
        row = dict(alt)
        row["transfers_out"] = enrich_rows(list(row.get("transfers_out") or []), by_id)
        row["transfers_in"] = enrich_rows(list(row.get("transfers_in") or []), by_id)
        alts.append(row)
    out["alternatives"] = alts
    out["xi"] = enrich_rows(list(out.get("xi") or []), by_id)
    for key in ("captain_hold", "captain_after"):
        row = out.get(key) or {}
        eid = _eid(row)
        out[key] = merge_context(row, by_id.get(eid) if eid is not None else None)
    return out


def enrich_recap(recap: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out = dict(recap)
    out["players"] = enrich_rows(list(out.get("players") or []), by_id)
    return out


def roster_from_ids(ids: list[int], by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in ids:
        eid = int(raw)
        ctx = by_id.get(eid, {"element_id": eid})
        rows.append(dict(ctx))
    return rows


def format_briefing(
    recap: dict[str, Any],
    upcoming: dict[str, Any] | None,
    *,
    flags: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# Weekly briefing — {upcoming['season'] if upcoming else recap.get('season')} "
        f"GW{upcoming['event'] if upcoming else recap.get('event')}",
        "",
        "## Previous gameweek",
        recap.get("headline") or f"{recap.get('season')} GW{recap.get('event')}",
    ]
    if recap.get("note"):
        lines.append(recap["note"])
    if recap.get("players"):
        lines.append(
            f"Combined from {recap['n_found']} matched players: **{recap['total_points']:.0f} pts**."
        )
        if recap.get("best"):
            b = recap["best"]
            lines.append(f"Best: {b['name']} {b['points']:.0f} pts ({b['minutes']:.0f}').")
        if recap.get("did_not_play"):
            lines.append("Did not play: " + ", ".join(recap["did_not_play"]) + ".")
        elif recap.get("blanks"):
            lines.append("Blanks (0 pts): " + ", ".join(recap["blanks"]) + ".")
        lines.append("")
        lines.append("| Player | Position | Club | Minutes | Points | Form | Next fixtures |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for p in recap["players"]:
            form = p.get("form")
            form_s = f"{float(form):.1f}" if isinstance(form, (int, float)) else "—"
            nxt = p.get("next_5_short") or "—"
            club = p.get("team") or "—"
            lines.append(
                f"| {p['name']} | {p['position']} | {club} | {p['minutes']:.0f} | "
                f"{p['points']:.0f} | {form_s} | {nxt} |"
            )
    else:
        lines.append("No recap rows available for this 15.")
    if recap.get("missing_ids"):
        lines.append(f"Unmatched ids: {recap['missing_ids']}")

    lines.extend(["", "## What I'd do this week"])
    if upcoming is None:
        lines.append("Save a legal 15 first, then generate the briefing.")
        return "\n".join(lines)

    if upcoming["action"] == "HOLD":
        lines.append(take_argument(upcoming, recap, flags or []))
        cap = upcoming["captain_hold"]
        lines.append(
            f"I'd give the armband to {cap['name']}. "
            f"I have him down for about {float(cap['xpts']):.1f} points this week and he looks like starting."
        )
        if upcoming["chip_beats_no_chip"]:
            lines.append(
                f"A chip wins the arithmetic this week ({upcoming['chip_best_label']}). "
                f"{upcoming['chip_note']}"
            )
        else:
            lines.append(
                f"I would not play a chip — do not play one because a one-week table looks good. "
                f"{upcoming['chip_note']}"
            )
    else:
        lines.append(take_argument(upcoming, recap, flags or []))
    if upcoming.get("engine_note"):
        lines.extend(["", upcoming["engine_note"]])
    if flags:
        lines.extend(["", "## Official FPL flags on your 15"])
        for row in flags:
            news = row.get("news") or ""
            lines.append(f"- {row['name']} ({row.get('status', 'a')}): {news or 'no news'}")
    lines.extend(
        [
            "",
            "I do not scrape journalism or guess XIs. Minutes overrides go in `data/overrides/xmins.csv`.",
        ]
    )
    return "\n".join(lines)


def squad_flags(players: pd.DataFrame, squad_ids: list[int]) -> list[dict[str, Any]]:
    if players is None or players.empty:
        return []
    work = players.copy()
    work["element_id"] = pd.to_numeric(work["element_id"], errors="coerce")
    work = work.loc[work["element_id"].isin([int(i) for i in squad_ids])]
    flags: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        status = str(row.get("status") or "a").lower()
        news = str(row.get("news") or "").strip()
        chance = row.get("chance_of_playing_next_round")
        if status in {"a", ""} and not news:
            continue
        flags.append(
            {
                "element_id": int(row["element_id"]),
                "name": str(row.get("web_name") or row.get("name") or row["element_id"]),
                "status": status,
                "news": news,
                "chance_of_playing_next_round": None if pd.isna(chance) else float(chance),
            }
        )
    return flags


def build_weekly_briefing(
    *,
    settings: Settings | None = None,
    season: str | None = None,
    event: int | None = None,
    squad_ids: list[int],
    free_transfers: int = 1,
) -> dict[str, Any]:
    if len({int(i) for i in squad_ids}) != SQUAD_SIZE:
        raise RuntimeError(f"Need {SQUAD_SIZE} unique ids for a briefing")
    cfg = settings or load_settings()
    panel_path = cfg.processed_dir / "player_gw.parquet"
    if not panel_path.exists():
        raise RuntimeError("Missing player_gw.parquet. Run `python -m fpl history` first.")
    panel = pd.read_parquet(panel_path)
    ids = [int(i) for i in squad_ids]
    chip_result = run_chips(
        settings=cfg,
        season=season,
        event=event,
        squad_ids=ids,
        free_transfers=free_transfers,
    )
    upcoming = upcoming_facts_from_chips(chip_result)
    recap = build_last_gw_recap(
        settings=cfg,
        panel=panel,
        squad_ids=ids,
        current_season=str(upcoming["season"] or cfg.current_season),
        upcoming_event=int(upcoming["event"] or 1),
    )
    by_id = load_player_context(cfg, from_event=int(upcoming["event"] or 1))
    upcoming = enrich_upcoming(upcoming, by_id)
    recap = enrich_recap(recap, by_id)
    roster = roster_from_ids(ids, by_id)
    players_path = cfg.processed_dir / "players.parquet"
    flag_ids = list(ids)
    for bucket in (upcoming.get("transfers_in") or [], upcoming.get("transfers_out") or []):
        for row in bucket:
            try:
                flag_ids.append(int(row["element_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    flags = (
        squad_flags(pd.read_parquet(players_path), list(dict.fromkeys(flag_ids)))
        if players_path.exists()
        else []
    )
    markdown = format_briefing(recap, upcoming, flags=flags)
    facts = {
        "recap": recap,
        "upcoming": upcoming,
        "flags": flags,
        "squad_ids": ids,
        "roster": roster,
    }
    return {
        "markdown": markdown,
        "facts": facts,
        "eval_path": cfg.eval_dir / "briefing.md",
    }


def write_briefing(result: dict[str, Any], settings: Settings | None = None) -> Path:
    cfg = settings or load_settings()
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.eval_dir / "briefing.md"
    path.write_text(result["markdown"], encoding="utf-8")
    return path
