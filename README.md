# FPL analyser

Local expected-points engine (CLI + a basic local app). **How it works and how to use it:** [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Open the local app (browser on your machine):

```powershell
python -m fpl app
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

## Expected points

Structural FPL scoring (appearance, xG, CS Poisson, BPS/bonus). Walk-forward vs last-5 and last-1. FPL `xP` is reported but labelled leaky. Shipped `xpts` is the structural model, not a blend with last-5.

```powershell
python -m fpl xpts
```

Outputs: `data/processed/xpts_oos.parquet`, `data/processed/eval/xpts.json`

## Squad builder

Single-GW PuLP: 15-man squad, £100.0m, 3-per-club, legal XI, captain and vice. Objective is risk-adjusted (unconditional) `xPts`; captain is 2× that, but only eligible if `p_play` clears a gate (default 0.75).

```powershell
python -m fpl squad                 # upcoming GW (live prices + last completed PL form)
python -m fpl squad --season 2025-26 --event 38   # historical backtest
```

Outputs: `data/processed/squad.parquet`, `data/processed/eval/squad.json`

Kill criterion: every solution is legal, and on a toy pool the solver matches brute force — including refusing to captain a high-mean rotation risk.

## Transfers

Myopic 0–3 transfers from a current 15. Hits cost 4 points. Hold (re-pick XI/captain only) is always the baseline. Wildcard is a Phase 4 rebuild (0 hits).

```powershell
python -m fpl transfer --squad data/overrides/squad.csv --free-transfers 1
python -m fpl transfer --season 2025-26 --event 20
python -m fpl transfer --wildcard --squad data/overrides/squad.csv
python -m fpl transfer --backtest
```

Without `--squad` / `--team-id` on a historical GW, the current 15 is last GW's solver squad (a backtest device). On the live season you must pass your 15.

Outputs: `data/processed/transfers.parquet`, `data/processed/eval/transfers.json` (or `transfers_backtest.json`).

Kill criterion: a recommended hit has positive expected net vs doing nothing.

## Chips

This-GW expected points for Bench Boost, Triple Captain, and Free Hit / Wildcard versus hold and ordinary transfers. One chip per week. v1 does **not** auto-play a chip — saving TC for a nailed premium DGW is usually right even if this week’s extra copy looks fine.

```powershell
python -m fpl chips --season 2025-26 --event 20
python -m fpl chips --squad data/overrides/squad.csv
```

FH and WC share this-GW EV; they differ only in whether you keep the squad. BB after transfers is often weaker than BB on the hold 15 because the solver dumps enablers onto the bench.

Output: `data/processed/eval/chips.json`

## Advisor

Last completed GW recap for your 15, then this week's HOLD/TAKE, captain, and chip EV — the engine in English, plus a chat that answers the question you asked (one named player is one swap, not a dump of the whole TAKE).

```powershell
python -m fpl brief --squad data/overrides/squad.csv
python -m fpl app
```

In the app: save your 15, then **Generate this week's briefing**. Optional `OPENAI_API_KEY` only rephrases; it must not invent news.
