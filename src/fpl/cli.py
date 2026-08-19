from __future__ import annotations

import argparse
import sys

from fpl.ingest.client import FplApiError
from fpl.ingest.history import format_history_report, run_history
from fpl.ingest.pipeline import format_ingest_report, run_ingest
from fpl.models.pipeline import format_minutes_report, format_xpts_report, run_minutes, run_xpts
from fpl.optimize.chips import format_chip_report, run_chips
from fpl.optimize.squad import format_squad_report, run_squad
from fpl.optimize.transfers import format_backtest_report, format_transfer_report, run_transfer, run_transfer_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpl", description="FPL analyser CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Fetch and cache official FPL bootstrap + fixtures")
    ingest.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore a fresh on-disk snapshot and hit the FPL API again",
    )
    history = sub.add_parser(
        "history",
        help="Build the as-of player-gameweek panel from vaastav (+ live element-summary)",
    )
    history.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download vaastav CSVs and element-summary JSON",
    )
    history.add_argument(
        "--no-current",
        action="store_true",
        help="Skip live element-summary for the current season",
    )
    minutes = sub.add_parser(
        "minutes",
        help="Walk-forward minutes/rotation model vs rolling baselines",
    )
    xpts = sub.add_parser(
        "xpts",
        help="Walk-forward expected points vs last-5 points and FPL xP",
    )
    squad = sub.add_parser(
        "squad",
        help="Single-GW ILP: 15-man squad, XI, captain (risk-adjusted xPts)",
    )
    squad.add_argument(
        "--season",
        help="Season key, e.g. 2026-27 (default: live current season, else latest in player_gw)",
    )
    squad.add_argument("--event", type=int, help="Gameweek number (default: next unfinished fixture)")
    squad.add_argument("--budget", type=float, default=100.0, help="Budget in millions (default 100.0)")
    squad.add_argument(
        "--captain-p-play",
        type=float,
        default=0.75,
        help="Minimum p_play to be eligible for the armband (default 0.75)",
    )
    squad.add_argument(
        "--live-prices",
        action="store_true",
        help="Overlay current bootstrap prices when the season is the live one",
    )
    transfer = sub.add_parser(
        "transfer",
        help="Myopic transfers from a current 15: 0–3 moves, 4-pt hits vs hold",
    )
    transfer.add_argument("--season", help="Season key, e.g. 2025-26")
    transfer.add_argument("--event", type=int, help="Gameweek number")
    transfer.add_argument("--squad", help="CSV of current 15 element_ids")
    transfer.add_argument("--team-id", type=int, help="FPL entry id (loads official picks)")
    transfer.add_argument("--bank", type=float, help="ITB in millions (default: 100 minus current cost)")
    transfer.add_argument("--free-transfers", type=int, default=1)
    transfer.add_argument("--max-transfers", type=int, default=3)
    transfer.add_argument("--captain-p-play", type=float, default=0.75)
    transfer.add_argument("--wildcard", action="store_true", help="Rebuild the 15 (Phase 4, 0 hits)")
    transfer.add_argument("--backtest", action="store_true", help="Walk selected GWs: prev optimal → transfers")
    transfer.add_argument(
        "--live-prices",
        action="store_true",
        help="Overlay current bootstrap prices when the season is the live one",
    )
    chips = sub.add_parser(
        "chips",
        help="This-GW EV for BB, TC, FH/WC vs hold and transfers (no auto-play)",
    )
    chips.add_argument("--season", help="Season key, e.g. 2025-26")
    chips.add_argument("--event", type=int, help="Gameweek number")
    chips.add_argument("--squad", help="CSV of current 15 element_ids")
    chips.add_argument("--team-id", type=int, help="FPL entry id")
    chips.add_argument("--bank", type=float, help="ITB in millions")
    chips.add_argument("--free-transfers", type=int, default=1)
    chips.add_argument("--max-transfers", type=int, default=3)
    chips.add_argument("--captain-p-play", type=float, default=0.75)
    chips.add_argument("--live-prices", action="store_true")
    brief = sub.add_parser(
        "brief",
        help="Last-GW recap + this-GW HOLD/TAKE briefing for your 15",
    )
    brief.add_argument("--season", help="Season key (default: live current)")
    brief.add_argument("--event", type=int, help="Gameweek (default: next unfinished)")
    brief.add_argument("--squad", help="CSV of current 15 element_ids")
    brief.add_argument("--free-transfers", type=int, default=1)
    sub.add_parser("app", help="Open the local web UI (Streamlit)")
    return parser


def _element_progress(index: int, total: int, element_id: int) -> None:
    if index == 1 or index == total or index % 50 == 0:
        print(f"  element-summary {index}/{total} (id={element_id})", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        try:
            result = run_ingest(refresh=args.refresh)
        except FplApiError as exc:
            print(f"Ingest failed: {exc}", file=sys.stderr)
            return 1
        print(format_ingest_report(result))
        return 0
    if args.command == "history":
        try:
            result = run_history(
                refresh=args.refresh,
                include_current=not args.no_current,
                on_progress=_element_progress,
            )
        except (FplApiError, RuntimeError, OSError) as exc:
            print(f"History failed: {exc}", file=sys.stderr)
            return 1
        print(format_history_report(result))
        return 0
    if args.command == "minutes":
        try:
            result = run_minutes()
        except (RuntimeError, OSError) as exc:
            print(f"Minutes model failed: {exc}", file=sys.stderr)
            return 1
        print(format_minutes_report(result))
        return 0
    if args.command == "xpts":
        try:
            result = run_xpts()
        except (RuntimeError, OSError) as exc:
            print(f"xPts failed: {exc}", file=sys.stderr)
            return 1
        print(format_xpts_report(result))
        return 0
    if args.command == "squad":
        try:
            result = run_squad(
                season=args.season,
                event=args.event,
                budget_m=args.budget,
                captain_min_p_play=args.captain_p_play,
                overlay_live=args.live_prices,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Squad solver failed: {exc}", file=sys.stderr)
            return 1
        print(format_squad_report(result["solution"]))
        if result.get("note"):
            print(result["note"])
        print(f"Eval file: {result['eval_path']}")
        return 0
    if args.command == "transfer":
        try:
            if args.backtest:
                result = run_transfer_backtest()
                print(format_backtest_report(result))
                return 0
            result = run_transfer(
                season=args.season,
                event=args.event,
                squad_path=args.squad,
                team_id=args.team_id,
                bank_m=args.bank,
                free_transfers=args.free_transfers,
                max_transfers=args.max_transfers,
                captain_min_p_play=args.captain_p_play,
                wildcard=args.wildcard,
                overlay_live=args.live_prices,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Transfer solver failed: {exc}", file=sys.stderr)
            return 1
        print(format_transfer_report(result["plan"]))
        if result.get("note"):
            print(result["note"])
        print(f"Eval file: {result['eval_path']}")
        return 0
    if args.command == "chips":
        try:
            result = run_chips(
                season=args.season,
                event=args.event,
                squad_path=args.squad,
                team_id=args.team_id,
                bank_m=args.bank,
                free_transfers=args.free_transfers,
                max_transfers=args.max_transfers,
                captain_min_p_play=args.captain_p_play,
                overlay_live=args.live_prices,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Chip EV failed: {exc}", file=sys.stderr)
            return 1
        print(format_chip_report(result["report"]))
        if result.get("note"):
            print(result["note"])
        print(f"Eval file: {result['eval_path']}")
        return 0
    if args.command == "brief":
        from pathlib import Path

        from fpl.advisor.briefing import build_weekly_briefing, write_briefing
        from fpl.config import load_settings
        from fpl.optimize.transfers import load_squad_csv

        cfg = load_settings()
        squad_path = Path(args.squad) if args.squad else cfg.overrides_dir / "squad.csv"
        try:
            ids = load_squad_csv(squad_path)
            result = build_weekly_briefing(
                settings=cfg,
                season=args.season,
                event=args.event,
                squad_ids=ids,
                free_transfers=args.free_transfers,
            )
            path = write_briefing(result, settings=cfg)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Briefing failed: {exc}", file=sys.stderr)
            return 1
        print(result["markdown"])
        print(f"\nWrote {path}")
        return 0
    if args.command == "app":
        from fpl.ui.launch import launch

        return launch()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
