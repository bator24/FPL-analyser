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
    vaastav_base_url: str
    history_seasons: tuple[str, ...]
    current_season: str
    element_summary_delay: float

    @property
    def snapshots_dir(self) -> Path:
        return self.raw_dir / "snapshots"

    @property
    def vaastav_dir(self) -> Path:
        return self.raw_dir / "vaastav"

    @property
    def element_summary_dir(self) -> Path:
        return self.raw_dir / "element-summary"

    @property
    def eval_dir(self) -> Path:
        return self.processed_dir / "eval"

    @property
    def models_dir(self) -> Path:
        return self.processed_dir / "models"


def load_settings(config_path: Path | None = None) -> Settings:
    root = project_root()
    path = config_path or (root / "configs" / "default.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fpl = raw.get("fpl") or {}
    data = raw.get("data") or {}
    cache = raw.get("cache") or {}
    history = raw.get("history") or {}
    seasons = history.get("seasons") or [
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]
    return Settings(
        base_url=str(fpl.get("base_url", "https://fantasy.premierleague.com/api")).rstrip("/"),
        timeout_seconds=int(fpl.get("timeout_seconds", 30)),
        user_agent=str(fpl.get("user_agent", "FPLAnalyser/0.1")),
        raw_dir=resolve_under_root(data.get("raw_dir", "data/raw"), root=root),
        processed_dir=resolve_under_root(data.get("processed_dir", "data/processed"), root=root),
        overrides_dir=resolve_under_root(data.get("overrides_dir", "data/overrides"), root=root),
        ttl_hours=float(cache.get("ttl_hours", 6)),
        root=root,
        vaastav_base_url=str(
            history.get(
                "vaastav_base_url",
                "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data",
            )
        ).rstrip("/"),
        history_seasons=tuple(str(s) for s in seasons),
        current_season=str(history.get("current_season", "2026-27")),
        element_summary_delay=float(history.get("element_summary_delay", 0.12)),
    )
