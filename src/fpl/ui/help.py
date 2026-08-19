"""In-app how-to copy. Keep this in lockstep with the Streamlit layout."""

HOWTO_TITLE = "How to use this app"

HOWTO_BODY = """
This is a local expected-points tool for **your 15 this gameweek**. It does not log into FPL for you, does not play chips, and does not scrape news.

### First time (or after a long break)

1. Open the **left sidebar**.
2. Click **Ingest FPL snapshot** (prices, fixtures, availability).
3. Click **Rebuild history panel** once. That can take several minutes. Reload the page when it finishes.
4. Leave **Season** on the live season and **Gameweek** on the next unfinished GW unless you are backtesting.

### Every week (the actual loop)

1. Set **Free transfers** in the sidebar (usually 1).
2. Build **your 15** on the green pitch — or type your FPL team ID and **Load picks from FPL**.
3. Click **Save squad**. It stays disabled until you have 2 GKP / 5 DEF / 5 MID / 3 FWD, **£100.0m or less**, and at most 3 from one club.
4. Click **Recommend transfers**. Read **TAKE** or **HOLD**. TAKE is only suggested if expected points beat doing nothing *after* 4-point hits.
5. Optionally click **Chip EV**. A starred line is this-week arithmetic — do **not** blindly play Triple Captain.
6. Optionally click **Generate this week's briefing**, then ask the advisor why (captain, last GW, that transfer). It only uses engine numbers, not Twitter.

**Rebuild 15 from scratch** ignores your team. Use it as a wildcard-shaped comparison, not as “my squad”.

### Pitch

- Click **Remove** under a filled shirt (or **Remove {name}** in the picker) to take them off. Clicking the shirt only selects it to replace.
- **Type a name in Search** (Salah, Fernandes, …). That box searches first/last/`web_name`, not the shirt you clicked. The **Choose** dropdown is also type-to-filter by those labels.
- Filter further by club, price, total points; sort as you like.
- Early in the season, **Pts** are often 0 — sort by price or form instead.
- Injured / doubtful flags come from official FPL `news`, not a scrape. If a presser changes minutes, that still belongs in `data/overrides/xmins.csv` (not in this screen yet).

### How to read the numbers

- **xPts** = expected FPL points if you field them, already haircut by the chance they blank.
- **p_play** = chance they play. Captain must be ≥ 0.75 when someone nailed exists.
- Last-GW recap in the advisor is raw combined points (no captain, no auto-subs). Until 2026/27 histories exist, it uses last season’s finale mapped onto this 15.
"""

SIDEBAR_STEPS = """
1. Ingest (if prices look stale)  
2. Set free transfers  
3. Pitch → Save squad  
4. Recommend transfers  
5. Chip EV if you might play a chip  
6. Advisor briefing to talk it through
"""
