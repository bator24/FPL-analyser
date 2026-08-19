from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from fpl.ingest.http import DownloadError, fetch_json

Json = dict[str, Any] | list[Any]
FetchFn = Callable[[str], Json]


def utc_now() -> datetime:
    return datetime.now(UTC)


def snapshot_id(moment: datetime | None = None) -> str:
    stamp = (moment or utc_now()).strftime("%Y-%m-%dT%H%M%SZ")
    return stamp


class FplApiError(RuntimeError):
    """Raised when the official FPL API cannot be read."""


def default_fetch_json(url: str, *, timeout: int, user_agent: str) -> Json:
    try:
        return fetch_json(url, timeout=timeout, user_agent=user_agent)
    except DownloadError as exc:
        raise FplApiError(str(exc)) from exc


class FplClient:
    """Fetches bootstrap-static and fixtures, caching raw JSON on disk."""

    def __init__(
        self,
        *,
        base_url: str,
        snapshots_dir: Path,
        ttl: timedelta,
        timeout_seconds: int = 30,
        user_agent: str = "FPLAnalyser/0.1",
        fetch_json: FetchFn | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.snapshots_dir = snapshots_dir
        self.ttl = ttl
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self._fetch_json = fetch_json

    def load_snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return bootstrap + fixtures. Hits the network only if cache is missing/stale or refresh=True."""
        latest = None if refresh else self._read_latest_meta()
        if latest is not None and not self._is_expired(latest):
            return self._read_snapshot_dir(self.snapshots_dir / latest["id"], from_cache=True)

        created = utc_now()
        sid = snapshot_id(created)
        dest = self.snapshots_dir / sid
        dest.mkdir(parents=True, exist_ok=True)

        bootstrap = self._get("bootstrap-static/")
        fixtures = self._get("fixtures/")
        (dest / "bootstrap-static.json").write_text(
            json.dumps(bootstrap, ensure_ascii=False), encoding="utf-8"
        )
        (dest / "fixtures.json").write_text(
            json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "id": sid,
            "created_at": created.isoformat(),
            "from_cache": False,
            "endpoints": ["bootstrap-static", "fixtures"],
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshots_dir / "latest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "id": sid,
            "created_at": created,
            "from_cache": False,
            "bootstrap": bootstrap,
            "fixtures": fixtures,
        }

    def _get(self, path: str) -> Json:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if self._fetch_json is not None:
            return self._fetch_json(url)
        return default_fetch_json(url, timeout=self.timeout_seconds, user_agent=self.user_agent)

    def _read_latest_meta(self) -> dict[str, Any] | None:
        pointer = self.snapshots_dir / "latest.json"
        if not pointer.exists():
            return None
        try:
            return json.loads(pointer.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _is_expired(self, meta: dict[str, Any]) -> bool:
        created_raw = meta.get("created_at")
        if not created_raw:
            return True
        created = datetime.fromisoformat(str(created_raw))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        dest = self.snapshots_dir / str(meta.get("id", ""))
        if not (dest / "bootstrap-static.json").exists() or not (dest / "fixtures.json").exists():
            return True
        return utc_now() - created > self.ttl

    def _read_snapshot_dir(self, dest: Path, *, from_cache: bool) -> dict[str, Any]:
        bootstrap = json.loads((dest / "bootstrap-static.json").read_text(encoding="utf-8"))
        fixtures = json.loads((dest / "fixtures.json").read_text(encoding="utf-8"))
        meta_path = dest / "meta.json"
        created = utc_now()
        sid = dest.name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sid = str(meta.get("id", sid))
            created_raw = meta.get("created_at")
            if created_raw:
                created = datetime.fromisoformat(str(created_raw))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
        return {
            "id": sid,
            "created_at": created,
            "from_cache": from_cache,
            "bootstrap": bootstrap,
            "fixtures": fixtures,
        }

    def get_element_summary(self, element_id: int) -> dict[str, Any]:
        payload = self._get(f"element-summary/{element_id}/")
        if not isinstance(payload, dict):
            raise FplApiError(f"Unexpected element-summary payload for {element_id}")
        return payload
