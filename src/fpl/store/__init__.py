from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_tables(tables: dict[str, pd.DataFrame], dest: Path) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, frame in tables.items():
        frame.to_parquet(dest / f"{name}.parquet", index=False)
        counts[name] = int(len(frame))
    return counts


def write_snapshot_manifest(dest: Path, payload: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "snapshot.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
