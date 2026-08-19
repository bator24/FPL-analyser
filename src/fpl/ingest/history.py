from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.config import Settings, load_settings
from fpl.features.rolling import build_asof_features
from fpl.ingest.client import FplClient
from fpl.ingest.element_summary import load_element_summaries
from fpl.ingest.panel import add_match_goals, attach_fdr, normalize_element_histories, normalize_merged_gw
from fpl.ingest.vaastav import download_vaastav_seasons
from fpl.store import write_tables


def _load_live_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    players_path = processed_dir / "players.parquet"
    teams_path = processed_dir / "teams.parquet"
    if not players_path.exists() or not teams_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_parquet(players_path), pd.read_parquet(teams_path)


def build_history_panel(
    *,
    settings: Settings,
    refresh: bool = False,
    include_current: bool = True,
    client: FplClient | None = None,
    fetch_bytes_fn=None,
    on_progress=None,
) -> dict[str, Any]:
    kwargs = {}
    if fetch_bytes_fn is not None:
        kwargs["fetch_bytes_fn"] = fetch_bytes_fn
    downloaded = download_vaastav_seasons(settings=settings, refresh=refresh, **kwargs)

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for season, files in downloaded.items():
        merged = files.get("gws/merged_gw.csv")
        if merged is None:
            missing.append(season)
            continue
        frame = normalize_merged_gw(merged, season, files.get("teams.csv"))
        frame = attach_fdr(frame, files.get("fixtures.csv"))
        frames.append(frame)

    current_rows = 0
    if include_current:
        players, teams = _load_live_tables(settings.processed_dir)
        if players.empty:
            missing.append(f"{settings.current_season} (run `python -m fpl ingest` first)")
        else:
            live_client = client or FplClient(
                base_url=settings.base_url,
                snapshots_dir=settings.snapshots_dir,
                ttl=timedelta(hours=settings.ttl_hours),
                timeout_seconds=settings.timeout_seconds,
                user_agent=settings.user_agent,
            )
            ids = [int(x) for x in players["element_id"].dropna().tolist()]
            summaries = load_element_summaries(
                ids,
                client=live_client,
                settings=settings,
                refresh=refresh,
                on_progress=on_progress,
            )
            current = normalize_element_histories(summaries, players, teams, settings.current_season)
            if not current.empty:
                fixtures_path = settings.processed_dir / "fixtures.parquet"
                if fixtures_path.exists():
                    fx = pd.read_parquet(fixtures_path)
                    keep = [c for c in ["fixture_id", "team_h_difficulty", "team_a_difficulty"] if c in fx.columns]
                    if "fixture_id" in keep:
                        current = current.merge(fx[keep], on="fixture_id", how="left")
                        home = current["was_home"].fillna(False).astype(bool)
                        current["fdr"] = current["team_h_difficulty"].where(home, current["team_a_difficulty"])
                        current = current.drop(columns=["team_h_difficulty", "team_a_difficulty"], errors="ignore")
                frames.append(current)
                current_rows = len(current)

    if not frames:
        raise RuntimeError("No player-GW rows could be built. Check vaastav downloads and ingest cache.")

    print(f"Building as-of features on {sum(len(f) for f in frames)} rows...", flush=True)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_match_goals(panel)
    featured = build_asof_features(panel)
    counts = write_tables(
        {"player_gw": featured},
        settings.processed_dir,
    )
    manifest = {
        "panel": "player_gw",
        "seasons": sorted(featured["season"].dropna().astype(str).unique().tolist()),
        "rows": counts["player_gw"],
        "current_season_rows": current_rows,
        "missing_seasons": missing,
        "note": "Rolling features are as-of prior matches only. fpl_xp_posthoc is not a pre-match feature.",
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    (settings.processed_dir / "panel.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "player_gw": featured,
        "counts": counts,
        "missing_seasons": missing,
        "current_season_rows": current_rows,
        "processed_dir": settings.processed_dir,
    }


def format_history_report(result: dict[str, Any], *, sample_n: int = 6) -> str:
    panel: pd.DataFrame = result["player_gw"]
    lines = [
        "Player-GW panel",
        f"Processed: {result['processed_dir'] / 'player_gw.parquet'}",
        f"Rows: {result['counts']['player_gw']}",
        f"Seasons: {', '.join(sorted(panel['season'].dropna().astype(str).unique()))}",
        f"Current-season rows: {result['current_season_rows']}",
    ]
    if result["missing_seasons"]:
        lines.append(f"Missing: {', '.join(result['missing_seasons'])}")
    sample = panel.sort_values(["season", "event", "total_points"], ascending=[False, False, False]).head(sample_n)
    lines.append("\nRecent high-scoring rows (outcomes, not predictions):")
    for row in sample.itertuples(index=False):
        lines.append(
            f"  {row.season} GW{int(row.event) if pd.notna(row.event) else '?'} "
            f"{row.name} {row.position}  {int(row.minutes) if pd.notna(row.minutes) else 0}m  "
            f"{row.total_points}pts  minutes_r5={getattr(row, 'minutes_r5', float('nan'))}"
        )
    return "\n".join(lines)


def run_history(
    *,
    refresh: bool = False,
    include_current: bool = True,
    settings: Settings | None = None,
    on_progress=None,
) -> dict[str, Any]:
    cfg = settings or load_settings()
    return build_history_panel(
        settings=cfg,
        refresh=refresh,
        include_current=include_current,
        on_progress=on_progress,
    )
