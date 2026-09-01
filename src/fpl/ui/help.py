"""In-app how-to copy. Keep this in lockstep with the Streamlit layout."""

HOWTO_TITLE = "How to use this app"

HOWTO_BODY = """
This is a local expected-points tool for **your 15 this gameweek**. It does not log into FPL for you, does not play chips, and does not scrape news.

**Save squad** writes `data/overrides/squad.csv` on this computer. That file survives closing the browser and restarting Streamlit. Unsaved shirts live only in this session. Nothing is sent to FPL.

### First time (or after a long break)

1. Open the **left sidebar**.
2. Click **Ingest FPL snapshot** (prices, fixtures, availability).
3. Click **Rebuild history panel** once. That can take several minutes. Reload the page when it finishes.
4. Leave **Season** on the live season and **Gameweek** on the next unfinished GW unless you are backtesting.

### Every week (the actual loop)

1. Set **Free transfers** in the sidebar (usually 1).
2. On **My Team**, build your 15 on the pitch — or type your FPL team ID and **Load picks from FPL**.
3. Click **Save squad**. It stays disabled until you have 2 GKP / 5 DEF / 5 MID / 3 FWD, **£100.0m or less**, and at most 3 from one club.
4. Open **Transfers**. Read **TAKE** or **HOLD**. TAKE is only suggested if expected points beat doing nothing *after* 4-point hits. A two-move TAKE is a bundle — **Other ideas** lists several legal singles (and a few other two-move packages) you can pick instead, and will say if a half does not fit your bank on its own.
5. Optionally open **Chips**. A starred line is this-week arithmetic — do **not** blindly play Triple Captain.
6. Optionally open **Advisor**, generate the briefing, then ask **why Wilson** (or whoever). Named questions get that one move — expected points, whether they start, form, flags, next five, and whether you can take it without the rest of a two-move bundle. Asking “what should I do this week” still gets the full TAKE. No spreadsheet codes. It only uses engine numbers, not Twitter.

**Wildcard 15** ignores your team. Use it as a comparison, not as “my squad”.

### Pitch

- Click **Remove** under a filled shirt (or **Remove {name}** in the picker) to take them off. Clicking the shirt only selects it to replace. Hover a filled shirt for club, value, FPL form, and next 5 fixtures (official FDR, 5=hardest).
- Player tables (picker, Transfers XI, out→in) use the same hover card. The out→in recommendation shows club, value, expected points, chance they play, form, and next 5 for **both** the sale and the player coming in.
- **Type a name in Search** (Salah, Fernandes, …). That box searches first/last/`web_name`, not the shirt you clicked. The **Choose** dropdown is also type-to-filter by those labels.
- Filter further by club, price, total points; sort as you like.
- Early in the season, **Pts** are often 0 — sort by price or form instead.
- Injured / doubtful flags come from official FPL `news`, not a scrape. A percentage in that news string haircuts expected minutes even if FPL left `chance_of_playing` blank. This season's minutes and xG from the bootstrap also mix into last year's form, so a haul or a blank after GW1 can move the TAKE. If a presser changes minutes beyond that, use `data/overrides/xmins.csv`.

### How to read the numbers

- **xPts** = expected FPL points if you field them, already haircut by the chance they blank.
- **p_play** = chance they play. Captain must be ≥ 0.75 when someone nailed exists.
- Last-GW recap in the advisor is raw combined points (no captain, no auto-subs). Until 2026/27 histories exist, it uses last season’s finale mapped onto this 15.
"""

SIDEBAR_STEPS = """
1. Ingest (if prices look stale)  
2. Set free transfers  
3. My Team → Save squad  
4. Transfers → TAKE vs HOLD  
5. Chips if you might play a chip  
6. Advisor to talk it through
"""
