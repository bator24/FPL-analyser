from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from fpl.config import Settings
from fpl.ingest.client import FplApiError, FplClient


def load_element_summaries(
    element_ids: list[int],
    *,
    client: FplClient,
    settings: Settings,
    refresh: bool = False,
    delay: float | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Load cached element-summary JSON, fetching missing ids with a polite delay."""
    cache_dir = settings.element_summary_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    wait = settings.element_summary_delay if delay is None else delay
    summaries: dict[int, dict[str, Any]] = {}
    total = len(element_ids)
    for index, element_id in enumerate(element_ids, start=1):
        path = cache_dir / f"{element_id}.json"
        if path.exists() and not refresh:
            summaries[element_id] = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                payload = client.get_element_summary(element_id)
            except FplApiError:
                if on_progress is not None:
                    on_progress(index, total, element_id)
                if wait:
                    time.sleep(wait)
                continue
            path.write_text(json.dumps(payload), encoding="utf-8")
            summaries[element_id] = payload
            if wait:
                time.sleep(wait)
        if on_progress is not None:
            on_progress(index, total, element_id)
    return summaries
