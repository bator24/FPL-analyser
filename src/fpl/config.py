from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fpl.paths import project_root, resolve_under_root


@dataclass(frozen=True)
class Settings:
    base_url: str
    timeout_seconds: int
    user_agent: str
    raw_dir: Path
    processed_dir: Path
    overrides_dir: Path
    ttl_hours: float
    root: Path

    @property
    def snapshots_dir(self) -> Path:
        return self.raw_dir / "snapshots"


def load_settings(config_path: Path | None = None) -> Settings:
    root = project_root()
    path = config_path or (root / "configs" / "default.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fpl = raw.get("fpl") or {}
    data = raw.get("data") or {}
    cache = raw.get("cache") or {}
    return Settings(
        base_url=str(fpl.get("base_url", "https://fantasy.premierleague.com/api")).rstrip("/"),
        timeout_seconds=int(fpl.get("timeout_seconds", 30)),
        user_agent=str(fpl.get("user_agent", "FPLAnalyser/0.1")),
        raw_dir=resolve_under_root(data.get("raw_dir", "data/raw"), root=root),
        processed_dir=resolve_under_root(data.get("processed_dir", "data/processed"), root=root),
        overrides_dir=resolve_under_root(data.get("overrides_dir", "data/overrides"), root=root),
        ttl_hours=float(cache.get("ttl_hours", 6)),
        root=root,
    )
