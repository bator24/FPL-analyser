"""Myopic transfer engine: 0–3 moves, 4-point hits, hold as the baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.config import Settings, load_settings
from fpl.ingest.client import FplApiError, default_fetch_json
from fpl.optimize.pool import (
    collapse_gameweek,
    default_season_event,
    overlay_live_prices,
    score_season,
)
from fpl.optimize.rules import (
    BUDGET_M,
    CAPTAIN_MIN_P_PLAY,
    FREE_TRANSFERS_DEFAULT,
    HIT_COST,
    MAX_TRANSFERS_DEFAULT,
    SQUAD_SIZE,
    normalize_position,
)
from fpl.optimize.squad import SquadSolution, format_squad_report, solve_squad


BACKTEST_SEASONS = ("2024-25", "2025-26")
BACKTEST_EVENTS = (10, 16, 22, 28, 34)


@dataclass
class TransferPlan:
    hold: SquadSolution
    chosen: SquadSolution
    wildcard: SquadSolution | None
    current_ids: set[int]
    transfers_out: list[dict[str, Any]]
    transfers_in: list[dict[str, Any]]
    free_transfers: int
    expected_net: float
    recommend: bool
    mode: str

    @property
    def n_transfers(self) -> int:
        return int(self.chosen.n_transfers)

    @property
    def hits(self) -> int:
        return int(self.chosen.hits)


def load_squad_csv(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if "element_id" not in frame.columns:
        raise RuntimeError(f"{path} needs an element_id column")
    ids = pd.to_numeric(frame["element_id"], errors="coerce").dropna().astype(int).tolist()
    ids = list(dict.fromkeys(ids))
    if len(ids) != SQUAD_SIZE:
        raise RuntimeError(f"{path} has {len(ids)} ids; need {SQUAD_SIZE}")
    return ids


def fetch_entry_picks(
    team_id: int,
    event: int,
    *,
    settings: Settings,
) -> tuple[list[int], float]:
    """Official FPL entry picks + bank (millions). Selling prices are not used in v1."""
    base = settings.base_url.rstrip("/")
    url = f"{base}/entry/{int(team_id)}/event/{int(event)}/picks/"
    try:
        payload = default_fetch_json(
            url,
            timeout=settings.timeout_seconds,
            user_agent=settings.user_agent,
        )
    except FplApiError as exc:
        raise RuntimeError(f"Could not load FPL team {team_id}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected FPL picks payload")
    picks = payload.get("picks") or []
    ids = [int(p["element"]) for p in picks if "element" in p]
    if len(ids) != SQUAD_SIZE:
        raise RuntimeError(f"FPL team {team_id} GW{event} returned {len(ids)} picks")
    history = payload.get("entry_history") or {}
    bank_tenths = pd.to_numeric(history.get("bank"), errors="coerce")
    bank_m = float(bank_tenths) / 10.0 if pd.notna(bank_tenths) else 0.0
    return ids, bank_m


def merge_current_into_pool(pool: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Ensure the current 15 exist this GW; missing players get xPts 0."""
    if current.empty:
        return pool
    cur = current.copy()
    cur["element_id"] = pd.to_numeric(cur["element_id"], errors="coerce")
    cur = cur.dropna(subset=["element_id"])
    cur["element_id"] = cur["element_id"].astype(int)
    have = set(pd.to_numeric(pool["element_id"], errors="coerce").dropna().astype(int))
    missing = cur[~cur["element_id"].isin(have)].copy()
    if missing.empty:
        return pool
    stubs = []
    for _, row in missing.iterrows():
        stubs.append(
            {
                "element_id": int(row["element_id"]),
                "name": row.get("name", str(row["element_id"])),
                "position": normalize_position(row["position"]),
                "team": row.get("team"),
                "team_id": row.get("team_id"),
                "cost_m": float(row.get("cost_m") or 0),
                "xpts": 0.0,
                "p_play": 0.0,
                "xpts_if_plays": 0.0,
                "total_points": 0.0,
            }
        )
    return pd.concat([pool, pd.DataFrame(stubs)], ignore_index=True)


def _rows_for_ids(table: pd.DataFrame, ids: set[int]) -> list[dict[str, Any]]:
    keep = [c for c in ["element_id", "name", "position", "cost_m", "xpts", "p_play"] if c in table.columns]
    subset = table[table["element_id"].isin(ids)]
    return subset[keep].to_dict(orient="records")


def realized_xi_points(solution: SquadSolution) -> float | None:
    table = solution.table
    if "total_points" not in table.columns:
        return None
    pts = pd.to_numeric(table["total_points"], errors="coerce")
    if pts.isna().all():
        return None
    xi = table["in_xi"].fillna(False)
    cap = table["is_captain"].fillna(False)
    return float(pts[xi].fillna(0).sum() + pts[cap].fillna(0).sum())


def available_budget(pool: pd.DataFrame, current_ids: set[int], bank_m: float | None) -> float:
    costs = pd.to_numeric(
        pool.loc[pool["element_id"].isin(current_ids), "cost_m"],
        errors="coerce",
    ).fillna(0)
    spent = float(costs.sum())
    if bank_m is None:
        bank_m = max(0.0, BUDGET_M - spent)
    return spent + float(bank_m)


def solve_transfers(
    pool: pd.DataFrame,
    current_ids: set[int] | list[int],
    *,
    bank_m: float | None = None,
    free_transfers: int = FREE_TRANSFERS_DEFAULT,
    max_transfers: int = MAX_TRANSFERS_DEFAULT,
    hit_cost: float = HIT_COST,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
    wildcard: bool = False,
    include_wildcard: bool = True,
    season: str | None = None,
    event: int | None = None,
    current_meta: pd.DataFrame | None = None,
) -> TransferPlan:
    current = {int(i) for i in current_ids}
    if len(current) != SQUAD_SIZE:
        raise RuntimeError(f"Current squad size {len(current)} != {SQUAD_SIZE}")
    work = merge_current_into_pool(pool, current_meta if current_meta is not None else pool)
    budget = available_budget(work, current, bank_m)

    hold = solve_squad(
        work,
        budget_m=max(budget, work.loc[work["element_id"].isin(current), "cost_m"].sum()),
        captain_min_p_play=captain_min_p_play,
        season=season,
        event=event,
        locked_ids=current,
    )
    wc: SquadSolution | None = None
    if wildcard or include_wildcard:
        wc = solve_squad(
            work,
            budget_m=budget,
            captain_min_p_play=captain_min_p_play,
            season=season,
            event=event,
        )
        wc.n_transfers = SQUAD_SIZE - len(current.intersection(wc.squad_ids))
        wc.hits = 0
        wc.hit_cost = 0.0

    if wildcard:
        chosen = wc if wc is not None else hold
        mode = "wildcard"
    else:
        chosen = solve_squad(
            work,
            budget_m=budget,
            captain_min_p_play=captain_min_p_play,
            season=season,
            event=event,
            current_ids=current,
            max_transfers=max_transfers,
            free_transfers=free_transfers,
            hit_cost=hit_cost,
        )
        mode = "myopic"

    out_ids = current - chosen.squad_ids
    in_ids = chosen.squad_ids - current
    names = pd.concat([hold.table, chosen.table, work], ignore_index=True).drop_duplicates("element_id")
    expected_net = float(chosen.net_objective - hold.objective)
    recommend = (not wildcard) and chosen.n_transfers > 0 and expected_net > 0.05
    if wildcard:
        recommend = expected_net > 0.05
    return TransferPlan(
        hold=hold,
        chosen=chosen,
        wildcard=wc if not wildcard else None,
        current_ids=current,
        transfers_out=_rows_for_ids(names, out_ids),
        transfers_in=_rows_for_ids(names, in_ids),
        free_transfers=int(free_transfers),
        expected_net=expected_net,
        recommend=recommend,
        mode=mode,
    )


def format_transfer_report(plan: TransferPlan) -> str:
    chosen = plan.chosen
    hold = plan.hold
    verdict = "TAKE" if plan.recommend else "HOLD"
    if plan.mode == "wildcard":
        verdict = "WILDCARD" if plan.recommend else "HOLD (wildcard no better)"
    lines = [
        f"Transfers ({plan.mode})",
        f"Season {chosen.season or '?'}  event {chosen.event if chosen.event is not None else '?'}",
        f"Free transfers {plan.free_transfers}  moves {plan.n_transfers}  hits {plan.hits} x {chosen.hit_cost:.0f}",
        f"Hold EV {hold.objective:.2f}",
        f"Chosen EV {chosen.objective:.2f}  net after hits {chosen.net_objective:.2f}  vs hold {plan.expected_net:+.2f}",
        f"Recommend: {verdict}",
        "",
    ]
    if plan.transfers_out or plan.transfers_in:
        lines.append("Out -> in")
        n = max(len(plan.transfers_out), len(plan.transfers_in))
        for i in range(n):
            left = plan.transfers_out[i] if i < len(plan.transfers_out) else {}
            right = plan.transfers_in[i] if i < len(plan.transfers_in) else {}
            lines.append(
                f"  {left.get('name', '-'):<22} {left.get('xpts', 0):5.2f}  ->  "
                f"{right.get('name', '-'):<22} {right.get('xpts', 0):5.2f}"
            )
        lines.append("")
    if plan.wildcard is not None and plan.mode != "wildcard":
        wc_net = plan.wildcard.net_objective - hold.objective
        lines.append(
            f"Wildcard EV {plan.wildcard.objective:.2f}  vs hold {wc_net:+.2f}  "
            f"(chip; not the default recommendation)"
        )
        lines.append("")
    lines.append(format_squad_report(chosen))
    lines.append("")
    lines.append(
        "v1 uses listed cost_m as both buy and sell price. "
        "A hit is recommended only if expected net vs holding the same 15 is positive."
    )
    return "\n".join(lines)


def plan_to_dict(plan: TransferPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "season": plan.chosen.season,
        "event": plan.chosen.event,
        "recommend": plan.recommend,
        "n_transfers": plan.n_transfers,
        "hits": plan.hits,
        "free_transfers": plan.free_transfers,
        "hold_ev": round(plan.hold.objective, 4),
        "chosen_ev": round(plan.chosen.objective, 4),
        "chosen_net": round(plan.chosen.net_objective, 4),
        "expected_net": round(plan.expected_net, 4),
        "wildcard_ev": round(plan.wildcard.objective, 4) if plan.wildcard is not None else None,
        "transfers_out": plan.transfers_out,
        "transfers_in": plan.transfers_in,
        "legal": plan.chosen.legal,
        "kill_pass": (plan.hits == 0) or (plan.expected_net > 0),
    }


def _current_from_prev_optimal(
    scored_season: pd.DataFrame,
    event: int,
    *,
    captain_min_p_play: float,
) -> SquadSolution:
    prev_event = int(event) - 1
    prev_pool = collapse_gameweek(
        scored_season[pd.to_numeric(scored_season["event"], errors="coerce") == prev_event]
    )
    if prev_pool.empty:
        raise RuntimeError(f"No rows for previous GW {prev_event}; pass --squad or use GW>=2")
    return solve_squad(prev_pool, captain_min_p_play=captain_min_p_play, event=prev_event)


def run_transfer(
    *,
    settings: Settings | None = None,
    season: str | None = None,
    event: int | None = None,
    squad_path: Path | None = None,
    team_id: int | None = None,
    from_prev_optimal: bool = False,
    bank_m: float | None = None,
    free_transfers: int = FREE_TRANSFERS_DEFAULT,
    max_transfers: int = MAX_TRANSFERS_DEFAULT,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
    wildcard: bool = False,
    overlay_live: bool = False,
) -> dict[str, Any]:
    cfg = settings or load_settings()
    panel_path = cfg.processed_dir / "player_gw.parquet"
    if not panel_path.exists():
        raise RuntimeError("Missing player_gw.parquet. Run `python -m fpl history` first.")
    panel = pd.read_parquet(panel_path)
    if season is None or event is None:
        d_season, d_event = default_season_event(panel)
        season = season or d_season
        event = event if event is not None else d_event
    print(f"Scoring {season} GW{event} for transfers...", flush=True)
    scored = score_season(panel, str(season))
    pool = collapse_gameweek(
        scored[pd.to_numeric(scored["event"], errors="coerce") == int(event)]
    )
    if pool.empty:
        raise RuntimeError(f"No player_gw rows for {season} GW{event}")
    players_path = cfg.processed_dir / "players.parquet"
    if overlay_live and players_path.exists() and str(season) == cfg.current_season:
        pool = overlay_live_prices(pool, pd.read_parquet(players_path))

    current_meta = None
    if squad_path is not None:
        current_ids = set(load_squad_csv(Path(squad_path)))
    elif team_id is not None:
        picks, api_bank = fetch_entry_picks(int(team_id), int(event), settings=cfg)
        current_ids = set(picks)
        if bank_m is None:
            bank_m = api_bank
    else:
        prev = _current_from_prev_optimal(scored, int(event), captain_min_p_play=captain_min_p_play)
        current_ids = prev.squad_ids
        current_meta = prev.table
        print(
            f"Current 15 = optimal squad from {season} GW{int(event) - 1} "
            "(pass --squad to use your team).",
            flush=True,
        )

    plan = solve_transfers(
        pool,
        current_ids,
        bank_m=bank_m,
        free_transfers=free_transfers,
        max_transfers=max_transfers,
        captain_min_p_play=captain_min_p_play,
        wildcard=wildcard,
        season=str(season),
        event=int(event),
        current_meta=current_meta,
    )
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cfg.eval_dir / "transfers.json"
    squad_path_out = cfg.processed_dir / "transfers.parquet"
    plan.chosen.table.to_parquet(squad_path_out, index=False)
    eval_path.write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")
    return {
        "plan": plan,
        "eval_path": eval_path,
        "squad_path": squad_path_out,
        "note": (
            "2026/27 histories are empty until GW1 is written; "
            "default slice is the latest season/event in player_gw."
            if str(season) != cfg.current_season
            else ""
        ),
    }


def run_transfer_backtest(
    *,
    settings: Settings | None = None,
    seasons: tuple[str, ...] = BACKTEST_SEASONS,
    events: tuple[int, ...] = BACKTEST_EVENTS,
    free_transfers: int = FREE_TRANSFERS_DEFAULT,
    max_transfers: int = MAX_TRANSFERS_DEFAULT,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
) -> dict[str, Any]:
    """Previous-GW optimal 15 → this-GW transfers. Kill: recommended hits have expected_net > 0."""
    cfg = settings or load_settings()
    panel_path = cfg.processed_dir / "player_gw.parquet"
    if not panel_path.exists():
        raise RuntimeError("Missing player_gw.parquet. Run `python -m fpl history` first.")
    panel = pd.read_parquet(panel_path)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        available = set(panel["season"].astype(str).unique())
        if season not in available:
            continue
        print(f"Backtest {season}...", flush=True)
        scored = score_season(panel, season)
        for event in events:
            gw = collapse_gameweek(
                scored[pd.to_numeric(scored["event"], errors="coerce") == int(event)]
            )
            prev = collapse_gameweek(
                scored[pd.to_numeric(scored["event"], errors="coerce") == int(event) - 1]
            )
            if gw.empty or prev.empty:
                continue
            try:
                prev_sol = solve_squad(prev, captain_min_p_play=captain_min_p_play, event=event - 1)
                plan = solve_transfers(
                    gw,
                    prev_sol.squad_ids,
                    free_transfers=free_transfers,
                    max_transfers=max_transfers,
                    captain_min_p_play=captain_min_p_play,
                    include_wildcard=False,
                    season=season,
                    event=event,
                    current_meta=prev_sol.table,
                )
            except RuntimeError as exc:
                rows.append(
                    {
                        "season": season,
                        "event": int(event),
                        "n_transfers": None,
                        "hits": 0,
                        "recommend": False,
                        "expected_net": 0.0,
                        "hold_ev": None,
                        "chosen_net": None,
                        "realized_net": None,
                        "legal": False,
                        "hit_expected_positive": True,
                        "error": str(exc),
                    }
                )
                continue
            hold_real = realized_xi_points(plan.hold)
            chosen_real = realized_xi_points(plan.chosen)
            realized_net = None
            if hold_real is not None and chosen_real is not None:
                realized_net = chosen_real - plan.hits * HIT_COST - hold_real
            rows.append(
                {
                    "season": season,
                    "event": int(event),
                    "n_transfers": plan.n_transfers,
                    "hits": plan.hits,
                    "recommend": plan.recommend,
                    "expected_net": round(plan.expected_net, 4),
                    "hold_ev": round(plan.hold.objective, 4),
                    "chosen_net": round(plan.chosen.net_objective, 4),
                    "realized_net": None if realized_net is None else round(realized_net, 4),
                    "legal": plan.chosen.legal,
                    "hit_expected_positive": (plan.hits == 0) or (plan.expected_net > 0),
                }
            )
    if not rows:
        raise RuntimeError("Backtest found no overlapping season/event rows.")
    frame = pd.DataFrame(rows)
    solved = frame[frame["hold_ev"].notna()] if "hold_ev" in frame.columns else frame
    hit_rows = solved[solved["hits"] > 0]
    kill_pass = bool(solved["hit_expected_positive"].all()) if len(solved) else False
    report = {
        "n": int(len(solved)),
        "n_errors": int(len(frame) - len(solved)),
        "n_recommend": int(solved["recommend"].sum()) if len(solved) else 0,
        "n_hits": int((solved["hits"] > 0).sum()) if len(solved) else 0,
        "mean_expected_net": float(solved["expected_net"].mean()) if len(solved) else None,
        "mean_expected_net_on_hits": float(hit_rows["expected_net"].mean()) if len(hit_rows) else None,
        "mean_realized_net": float(solved["realized_net"].mean())
        if len(solved) and solved["realized_net"].notna().any()
        else None,
        "kill_pass": kill_pass,
        "note": (
            "Kill is expected net vs hold, not realized. Realized is noisy single-GW luck. "
            "Current 15 is last GW's solver squad, not a human team."
        ),
        "rows": rows,
    }
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cfg.eval_dir / "transfers_backtest.json"
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report": report, "eval_path": eval_path}


def format_backtest_report(result: dict[str, Any]) -> str:
    report = result["report"]
    lines = [
        "Transfer backtest (prev-GW optimal → this GW, myopic)",
        f"GWs: {report['n']}  errors: {report.get('n_errors', 0)}  recommends: {report['n_recommend']}  hit weeks: {report['n_hits']}",
    ]
    if report.get("mean_expected_net") is not None:
        lines.append(f"Mean expected net vs hold: {report['mean_expected_net']:+.2f}")
    if report.get("mean_expected_net_on_hits") is not None:
        lines.append(f"Mean expected net on hit weeks: {report['mean_expected_net_on_hits']:+.2f}")
    if report.get("mean_realized_net") is not None:
        lines.append(f"Mean realized net vs hold: {report['mean_realized_net']:+.2f}  (noisy)")
    lines.extend(
        [
            f"Kill (recommended hits have expected_net > 0): {report['kill_pass']}",
            report.get("note", ""),
            f"Eval file: {result['eval_path']}",
        ]
    )
    return "\n".join(lines)
