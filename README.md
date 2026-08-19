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

## History panel

Builds a multi-season player-gameweek table with as-of rolling features (prior matches only). Uses the vaastav FPL dataset plus live `element-summary` for 2026/27.

```powershell
python -m fpl ingest
python -m fpl history
python -m fpl history --no-current
python -m fpl history --refresh
```

Output: `data/processed/player_gw.parquet`

`fpl_xp_posthoc` is vaastav's scraped FPL `xP` and is not a pre-match feature.

## Minutes model

Walk-forward by season. Two-stage: `P(play)` then `E[minutes | play]`. Always printed against last-3-GW and last-GW baselines.

```powershell
python -m fpl minutes
```

Outputs:

- `data/processed/minutes_oos.parquet`
- `data/processed/eval/minutes.json`
- `data/processed/models/minutes.joblib`

Kill criterion: beat rolling-3 MAE overall **and** on actual zeros.
