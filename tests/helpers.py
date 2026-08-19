from pathlib import Path

from fpl.config import Settings
from fpl.paths import project_root


def make_settings(tmp_path: Path, **overrides) -> Settings:
    root = project_root()
    values = dict(
        base_url="https://fantasy.premierleague.com/api",
        timeout_seconds=5,
        user_agent="test",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        overrides_dir=tmp_path / "overrides",
        ttl_hours=6,
        root=root,
        vaastav_base_url="https://example.test/vaastav",
        history_seasons=("2024-25",),
        current_season="2026-27",
        element_summary_delay=0.0,
    )
    values.update(overrides)
    return Settings(**values)
