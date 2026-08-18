from __future__ import annotations

import argparse
import sys

from fpl.ingest.client import FplApiError
from fpl.ingest.pipeline import format_ingest_report, run_ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpl", description="FPL analyser CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Fetch and cache official FPL bootstrap + fixtures")
    ingest.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore a fresh on-disk snapshot and hit the FPL API again",
    )
    return parser


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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
