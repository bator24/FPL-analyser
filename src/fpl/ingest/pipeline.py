from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from fpl.config import Settings, load_settings
from fpl.ingest.client import FplClient
from fpl.ingest.normalize import normalize
from fpl.store import write_snapshot_manifest, write_tables


def run_ingest(*, refresh: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or load_settings()
    client = FplClient(
        base_url=cfg.base_url,
        snapshots_dir=cfg.snapshots_dir,
        ttl=timedelta(hours=cfg.ttl_hours),
        timeout_seconds=cfg.timeout_seconds,
        user_agent=cfg.user_agent,
    )
    snapshot = client.load_snapshot(refresh=refresh)
    tables = normalize(snapshot["bootstrap"], snapshot["fixtures"])
    counts = write_tables(tables, cfg.processed_dir)
    write_snapshot_manifest(
        cfg.processed_dir,
        {
            "id": snapshot["id"],
            "created_at": snapshot["created_at"],
            "from_cache": snapshot["from_cache"],
            "counts": counts,
        },
    )
    return {
        "id": snapshot["id"],
        "from_cache": snapshot["from_cache"],
        "created_at": snapshot["created_at"],
        "counts": counts,
        "players": tables["players"],
        "teams": tables["teams"],
        "processed_dir": cfg.processed_dir,
        "snapshot_dir": cfg.snapshots_dir / snapshot["id"],
    }


def format_ingest_report(result: dict[str, Any], *, sample_n: int = 8) -> str:
    source = "cache" if result["from_cache"] else "FPL API"
    lines = [
        f"Snapshot {result['id']} ({source})",
        f"Raw: {result['snapshot_dir']}",
        f"Processed: {result['processed_dir']}",
        "",
        "Rows:",
    ]
    for name, count in result["counts"].items():
        lines.append(f"  {name}: {count}")

    players: pd.DataFrame = result["players"]
    teams: pd.DataFrame = result["teams"]
    if players.empty:
        lines.append("\nNo players in snapshot.")
        return "\n".join(lines)

    sample = players.merge(teams[["team_id", "short_name"]], on="team_id", how="left")
    sample = sample.sort_values(["ep_next", "total_points"], ascending=False).head(sample_n)
    lines.append("\nHighest FPL ep_next (not our model):")
    for row in sample.itertuples(index=False):
        news = f" | {row.news}" if isinstance(row.news, str) and row.news.strip() else ""
        lines.append(
            f"  {row.web_name:16} {row.short_name:3} {row.position}  "
            f"£{row.cost_m:.1f}  ep_next={row.ep_next}  status={row.status}{news}"
        )
    return "\n".join(lines)
