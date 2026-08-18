from datetime import timedelta
from pathlib import Path

from fpl.config import Settings
from fpl.ingest.client import FplClient
from fpl.ingest.pipeline import run_ingest
from fpl.paths import project_root


def _settings(tmp_path: Path) -> Settings:
    root = project_root()
    return Settings(
        base_url="https://fantasy.premierleague.com/api",
        timeout_seconds=5,
        user_agent="test",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        overrides_dir=tmp_path / "overrides",
        ttl_hours=6,
        root=root,
    )


def test_second_load_uses_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(url: str):
        calls.append(url)
        if url.endswith("bootstrap-static/"):
            return {
                "teams": [],
                "elements": [],
                "events": [],
                "element_types": [],
            }
        if url.endswith("fixtures/"):
            return []
        raise AssertionError(url)

    client = FplClient(
        base_url="https://fantasy.premierleague.com/api",
        snapshots_dir=tmp_path / "snapshots",
        ttl=timedelta(hours=6),
        fetch_json=fake_fetch,
    )
    first = client.load_snapshot(refresh=False)
    second = client.load_snapshot(refresh=False)
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert len(calls) == 2
    third = client.load_snapshot(refresh=True)
    assert third["from_cache"] is False
    assert len(calls) == 4


def test_run_ingest_writes_parquet(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch(url: str):
        if url.endswith("bootstrap-static/"):
            return {
                "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS", "code": 3}],
                "elements": [
                    {
                        "id": 1,
                        "web_name": "Saka",
                        "team": 1,
                        "element_type": 3,
                        "now_cost": 100,
                        "ep_next": "5.1",
                        "total_points": 0,
                        "status": "a",
                        "news": "",
                    }
                ],
                "events": [{"id": 1, "name": "Gameweek 1"}],
                "element_types": [{"id": 3, "singular_name_short": "MID"}],
            }
        return [{"id": 9, "event": 1, "team_h": 1, "team_a": 2}]

    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "fpl.ingest.pipeline.FplClient",
        lambda **kwargs: FplClient(
            base_url=settings.base_url,
            snapshots_dir=settings.snapshots_dir,
            ttl=timedelta(hours=settings.ttl_hours),
            fetch_json=fake_fetch,
        ),
    )
    result = run_ingest(refresh=False, settings=settings)
    assert result["counts"]["players"] == 1
    assert (settings.processed_dir / "players.parquet").exists()
    assert (settings.processed_dir / "snapshot.json").exists()
