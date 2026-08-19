# How the FPL analyser works (and how to use it)

This is a **local expected-points engine**, not an FPL app and not a mini-league rank maximizer. It estimates how many FPL points a player is worth *before* a gameweek, then builds a legal 15 / XI / captain, then compares transfers and chips against **doing nothing**.

There is no website. You run commands in PowerShell. That is intentional: the numbers are the product.

**Right now (August 2026):** history exists through **2025-26 GW38**. The 2026/27 `player_gw` panel is empty until FPL writes GW1 `element-summary` rows. Squad / transfer / chips still predict the **upcoming** gameweek: last completed PL form (mapped onto this year's players by stable FPL `code`) plus live prices, fixtures, and official FPL availability.

---

## What it optimizes

Expected points (EV) for *your* squad, this gameweek.

It does **not** optimize:

- Mini-league rank or differential ownership
- Price rises
- Multi-week chip calendars (it reports this-GW chip EV only)
- “Who FPL Review nailed this week”

A 50%-likely 8-point forward is treated as roughly **4 expected points**, not 8. Captaincy is gated: you cannot armband someone with `p_play` below 0.75 if a more nailed option exists.

---

## How the engine is built (pipeline)

```
Official FPL API + vaastav history
        ↓
Cached JSON / parquet snapshots
        ↓
player_gw table (one row per player per fixture, features as-of that GW)
        ↓
Minutes layer (will he play? how long?)
        ↓
xPts = FPL scoring rules × those minutes (plus CS + bonus)
        ↓
PuLP integer program: 15, XI, captain, vice, bench
        ↓
Transfers (0–3 moves, 4-point hits) vs hold
        ↓
Chips (BB / TC / FH / WC) as this-GW EV, never auto-played
```

### 1. Ingest (`python -m fpl ingest`)

Hits the official FPL API (`bootstrap-static`, `fixtures`), caches JSON for 6 hours, writes `data/processed/` tables (players, teams, events, fixtures). `--refresh` ignores a fresh cache.

### 2. History panel (`python -m fpl history`)

Builds `data/processed/player_gw.parquet` from:

- [vaastav](https://github.com/vaastav/Fantasy-Premier-League) merged gameweeks for 2020-21 … 2025-26
- Live `element-summary/{id}` for the current season (2026-27), when those histories exist

For the **next unfinished gameweek**, squad/transfer/chips do not wait for that row to exist. They take terminal form from the last completed matches (including last season's GW38 when 2026/27 is still empty), map players by FPL `code`, and attach this year's fixtures.

Rolling stats (1 / 3 / 5 / 10 / 38 matches) in the history panel are **shifted first, then rolled**. The current gameweek’s goals, minutes, and bonus are **targets**, never features. That is how backtests avoid leaking the result you are trying to predict. The live prior is the opposite on purpose: it *includes* the last completed match, because that match is already in the past.

`fpl_xp_posthoc` in the panel is vaastav’s scrape of FPL `xP`. It is **post-match / leaky**. It is reported in eval, not used as a pre-deadline input.

### 3. Minutes

Rotation is the hard problem. A gradient boosting minutes model **lost** to last-gameweek minutes on 2023–26 walk-forward, so shipped `e_minutes` is:

**last GW minutes → else last-3 mean → else 0**

`p_play` / `p_60` come from rolling “did he play / did he play 60+” rates. You can override a player-GW in `data/overrides/xmins.csv` after a presser. Stale overrides look like false confidence — delete them when the news is old.

### 4. Expected points (`xpts`)

Structural FPL scoring, not a neural net:

- Appearance (1 if plays, +1 if 60+)
- Goals / assists from rolling xG / xA (or realised goals if xG is missing), scaled to expected minutes
- Clean sheet: Poisson P(team concedes 0) × P(plays 60+)
- −1 per 2 goals conceded for GKP/DEF, saves for GKP, yellows
- Bonus from a BPS/bonus blend (not a full 22-player BPS sim)

Shipped `xpts` **is** that structural number. A blend with last-5 points was tried and **lost on MAE**. Walk-forward 2023–26: MAE 0.982 vs last-5 1.043; Spearman 0.693 vs 0.682. Haulers (≥5 points) are still slightly worse than last-5 — same shape as public models that lack a minutes edge.

### 5. Squad ILP (`python -m fpl squad`)

PuLP (CBC) picks a legal FPL 15:

- £100.0m, 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per club
- XI: 1 GKP, 3–5 DEF, 2–5 MID, 1–3 FWD
- Captain = 2× **unconditional** xPts, but only if `p_play ≥ 0.75`
- Vice is the next eligible player
- Bench is the leftover 4 (GK last in the printed order); auto-subs are **not** simulated yet

Objective for the XI is risk-adjusted (unconditional) xPts. The report also shows **naive-if-plays** (treat everyone as nailed) so you can see the minutes haircut.

`squad` **rebuilds from scratch**. It is a wildcard-shaped 15, not “your team plus two transfers.” When the current season has no `player_gw` rows yet, this is how you get a **balanced EV 15** for the upcoming GW (legal FPL constraints, not a vibe scout).

### 6. Transfers (`python -m fpl transfer`)

Starts from a **current 15**, then:

- **Hold:** same 15, re-pick XI and captain only
- **Myopic:** 0–3 transfers, 1 free transfer by default, extra moves cost **4 points** each
- Recommends a hit only if **expected net vs hold is positive**

Wildcard (`--wildcard`) is a free rebuild (Phase 4), 0 hits.

Without `--squad` or `--team-id`, the current 15 is **last GW’s solver squad**. That is a backtest device, not your FPL team. On the live season that path is disabled — pass `squad.csv` (this year’s `element_id`s) or `--team-id`.

v1 uses listed `cost_m` as both buy and sell price (not true FPL selling price).

### 7. Chips (`python -m fpl chips`)

One chip per week. This command **does not play a chip for you**. It prints this-GW EV for:

| Option | What it means |
|---|---|
| Hold | Re-pick XI/C only |
| Transfers | Ordinary 0–3 moves, hits deducted |
| Bench Boost on hold 15 | All 15 score; gain = bench xPts |
| Bench Boost after transfers | Same, on the post-transfer 15 (enabler benches often make this *worse*) |
| Triple Captain | One extra copy of the captain’s xPts (3× total) |
| Free Hit / Wildcard | Same this-GW rebuild EV. FH reverts next week; WC keeps the 15 |

Myopic “best this GW” can be a Triple Captain on a one-week spike. **Do not auto-play TC from that line.** Saving the chip for a nailed premium (especially a DGW) is usually right.

### 8. Advisor (`python -m fpl brief` / app section 3)

Not a generic FPL chatbot. It writes a **weekly briefing** from your 15:

- Last completed GW recap (or last season’s finale mapped by FPL `code` if 2026/27 has no rows yet)
- This GW: HOLD vs TAKE, captain, chip EV — the same ILP as `transfer` / `chips`
- Chat answers only from that briefing. Optional `OPENAI_API_KEY` rephrases; it still must not invent news or XIs

---

## How a user should use it

### One-time setup

```powershell
cd C:\FPL-analyser
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If the venv already exists (it does on this machine):

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)`. Then:

```powershell
python -m fpl ingest
python -m fpl history
```

### Local app (easiest)

Same engine, browser UI on your machine (nothing is uploaded):

```powershell
.\.venv\Scripts\Activate.ps1
python -m fpl app
```

A browser tab opens. Pick your 15 (search by name), save, then **Recommend transfers** or **Chip EV**. Season/gameweek and free transfers are in the sidebar.

You can still use the CLI below if you prefer.


You do **not** need `minutes` or `xpts` every week. Those rebuild walk-forward eval reports. Squad/transfer/chips score the next GW from `player_gw` when that row exists, otherwise from the live prior.

### Fill your team (required for real advice)

`data/overrides/squad.csv` must have **15 player IDs**, not your FPL entry ID.

```text
element_id
355
328
...
```

- **Player `element_id`:** the ID FPL uses for Haaland, Saka, etc. After ingest, they live in `data/processed/players.parquet` (`element_id`, `web_name`, `now_cost`).
- **Team / entry ID:** the number in `https://fantasy.premierleague.com/entry/1234567/`. You can pass that as `--team-id` instead of the CSV (loads official picks for that GW; needs network).

Until `squad.csv` has 15 rows, `transfer` / `chips` will not talk about *your* 15.

Optional: after team news, edit `data/overrides/xmins.csv` (see `data/overrides/README.md`). Blank rows are ignored.

### Weekly loop (once 2026/27 histories exist)

From the project folder, venv on:

```powershell
python -m fpl ingest --refresh
python -m fpl history
python -m fpl transfer --squad data/overrides/squad.csv --free-transfers 1
python -m fpl chips --squad data/overrides/squad.csv
```

Use `--free-transfers 2` if you rolled a FT. Use `--max-transfers 3` (default) unless you want to forbid hits.

Read **Recommend: TAKE / HOLD** on the transfer report first. Then glance at chips. Then look at the proposed XI; do not skip the haircut line.

### Dry run today (2025-26 only)

```powershell
python -m fpl squad --season 2025-26 --event 38
python -m fpl transfer --season 2025-26 --event 20
python -m fpl chips --season 2025-26 --event 20
```

These teach you the reports. They are not a 2026/27 pick.

### How to read the reports

**Squad / transfer XI lines**

- `xpts` — risk-adjusted (unconditional) expected points this GW  
- `if-plays` — same number divided by `p_play` (what you’d see if you ignored rotation)  
- `p_play` — estimated chance he plays at all  
- `C` / `V` — captain / vice  
- **haircut** — naive-if-plays minus risk-adjusted. Large haircut = the XI is full of rotation risk  

**Transfer**

- Hold EV vs chosen EV vs **net after hits**  
- TAKE only if net vs hold is clearly positive  
- Out → in is the move list  

**Chips**

- `*` marks best *this GW*  
- “Chip beats no-chip” is arithmetic, not an instruction to play the chip  

JSON copies of the last run live under `data/processed/eval/` (`squad.json`, `transfers.json`, `chips.json`, plus walk-forward `minutes.json` / `xpts.json` if you ran those).

---

## Commands (cheat sheet)

| Command | What it does |
|---|---|
| `python -m fpl ingest` | Cache live FPL bootstrap + fixtures |
| `python -m fpl history` | Rebuild `player_gw.parquet` |
| `python -m fpl minutes` | Walk-forward minutes eval (research) |
| `python -m fpl xpts` | Walk-forward xPts eval (research) |
| `python -m fpl squad` | Rebuild a legal 15 + XI + C from scratch |
| `python -m fpl transfer` | 0–3 transfers from a current 15 |
| `python -m fpl transfer --wildcard` | Chip: rebuild, keep the 15 |
| `python -m fpl transfer --backtest` | Hit-vs-hold check on historical GWs |
| `python -m fpl chips` | This-GW BB / TC / FH / WC vs hold |
| `python -m fpl brief` | Last-GW recap + this-GW HOLD/TAKE for your 15 |
| `python -m fpl app` | Local browser UI |

Useful flags: `--season 2025-26 --event 20`, `--squad data/overrides/squad.csv`, `--team-id 1234567`, `--free-transfers 1`, `--budget 100.0`, `--captain-p-play 0.75`.

---

## Honest limits

- **No cup / friendly / all-competitions form.** Understat and vaastav are Premier League. Extra competitions are a weak minutes signal and we do not scrape SofaScore/FotMob/news sites. Injured/doubtful comes from FPL `news` + `chance_of_playing_*`; presser minutes go in `data/overrides/xmins.csv`.
- **Live 2026/27 xPts is a prior**, not in-season evidence, until this GW is played. New signings without a last-season FPL `code` match are appearance-only (easy to underrate a Haaland-shaped arrival — override minutes if you are sure).
- Minutes are sticky last-GW, not FPL Review xMins. You are the lineup feed via `xmins.csv`.
- No auto-subs, no true selling prices, no price-change model, no 3–5 GW transfer lookahead.
- Chip table is **one-week EV**, not “should I save TC until GW8.”
- Haul prediction is not the engine’s strongest bucket.
- A web app would not fix any of the above; it would only wrap these commands.

When 2026-27 rows start appearing in `player_gw`, the live pool switches to **this season's** completed matches automatically. Fill `squad.csv` with **this year's** `element_id`s and use `transfer` / `chips` as the weekly pair.
