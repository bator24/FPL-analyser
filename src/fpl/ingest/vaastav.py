from __future__ import annotations

from pathlib import Path

from fpl.config import Settings
from fpl.ingest.http import DownloadError, fetch_bytes


SEASON_FILES = ("gws/merged_gw.csv", "fixtures.csv", "teams.csv")


def download_vaastav_season(
    season: str,
    *,
    settings: Settings,
    refresh: bool = False,
    fetch_bytes_fn=fetch_bytes,
) -> dict[str, Path | None]:
    """Cache vaastav CSVs for one season. Missing files become None, not an error."""
    dest_dir = settings.vaastav_dir / season
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path | None] = {}
    for rel in SEASON_FILES:
        local = dest_dir / Path(rel).name
        if local.exists() and not refresh:
            out[rel] = local
            continue
        url = f"{settings.vaastav_base_url}/{season}/{rel}"
        try:
            payload = fetch_bytes_fn(
                url, timeout=settings.timeout_seconds, user_agent=settings.user_agent
            )
        except DownloadError:
            out[rel] = None
            continue
        local.write_bytes(payload)
        out[rel] = local
    return out


def download_vaastav_seasons(
    seasons: tuple[str, ...] | None = None,
    *,
    settings: Settings,
    refresh: bool = False,
    fetch_bytes_fn=fetch_bytes,
) -> dict[str, dict[str, Path | None]]:
    chosen = seasons or settings.history_seasons
    return {
        season: download_vaastav_season(
            season, settings=settings, refresh=refresh, fetch_bytes_fn=fetch_bytes_fn
        )
        for season in chosen
    }
