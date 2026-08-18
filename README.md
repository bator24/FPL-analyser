# FPL analyser

Local expected-points engine. Phase 0 is ingest only.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Ingest

Pulls official `bootstrap-static` and `fixtures`, writes a timestamped JSON snapshot, and builds parquet tables.

```powershell
python -m fpl ingest
python -m fpl ingest          # second run uses cache if younger than 6 hours
python -m fpl ingest --refresh
```

Outputs:

- `data/raw/snapshots/<timestamp>/` raw JSON
- `data/processed/*.parquet` teams, players, events, fixtures

Minutes overrides (later phases): `data/overrides/xmins.csv`.
