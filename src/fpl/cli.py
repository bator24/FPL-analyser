from __future__ import annotations

import argparse
import sys

from fpl.ingest.client import FplApiError
from fpl.ingest.history import format_history_report, run_history
from fpl.ingest.pipeline import format_ingest_report, run_ingest
from fpl.models.pipeline import format_minutes_report, run_minutes


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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
