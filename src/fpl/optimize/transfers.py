"""Myopic transfer engine: 0–3 moves, 4-point hits, hold as the baseline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.config import Settings, load_settings
from fpl.ingest.client import FplApiError, default_fetch_json
from fpl.optimize.pool import (
    collapse_gameweek,
    load_prediction_pool,
    score_season,
)
from fpl.optimize.rules import (
    BUDGET_M,
    CAPTAIN_MIN_P_PLAY,
    FREE_TRANSFERS_DEFAULT,
    HIT_COST,
    MAX_PER_CLUB,
    MAX_TRANSFERS_DEFAULT,
    POSITION_ORDER,
    SQUAD_SIZE,
    club_key,
    legality_errors,
    normalize_position,
)
from fpl.optimize.squad import SquadSolution, format_squad_report, solve_squad


BACKTEST_SEASONS = ("2024-25", "2025-26")
BACKTEST_EVENTS = (10, 16, 22, 28, 34)
MENU_MARKET_PER_POS = 30
MENU_SINGLE_SCORE = 16
MENU_SINGLE_KEEP = 8
MENU_PAIR_SCORE = 6
MENU_PAIR_KEEP = 4
LOCKED_XI_TIME = 8


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
    alternatives: list[dict[str, Any]] = field(default_factory=list)

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


def _pair_by_position(
    outs: list[dict[str, Any]], ins: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    remaining = list(ins)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for out in outs:
        pos = str(out.get("position") or "")
        idx = next((i for i, inn in enumerate(remaining) if str(inn.get("position") or "") == pos), None)
        if idx is None:
            idx = 0 if remaining else None
        inn = remaining.pop(idx) if idx is not None else {}
        pairs.append((out, inn))
    return pairs


def _id_set(rows: list[dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for row in rows:
        try:
            out.add(int(row["element_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _option_fingerprint(out_ids: set[int], in_ids: set[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(sorted(out_ids)), tuple(sorted(in_ids)))


def _option_dict(
    *,
    key: str,
    label: str,
    n_transfers: int,
    hits: int,
    expected_net: float,
    transfers_out: list[dict[str, Any]],
    transfers_in: list[dict[str, Any]],
    legal: bool,
    note: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "n_transfers": int(n_transfers),
        "hits": int(hits),
        "expected_net": float(expected_net),
        "transfers_out": transfers_out,
        "transfers_in": transfers_in,
        "legal": bool(legal),
        "note": note,
    }


def _option_from_solution(
    current: set[int],
    hold: SquadSolution,
    sol: SquadSolution,
    names: pd.DataFrame,
    *,
    key: str,
    label: str,
    note: str = "",
) -> dict[str, Any]:
    out_ids = current - sol.squad_ids
    in_ids = sol.squad_ids - current
    return _option_dict(
        key=key,
        label=label,
        n_transfers=sol.n_transfers,
        hits=sol.hits,
        expected_net=float(sol.net_objective - hold.objective),
        transfers_out=_rows_for_ids(names, out_ids),
        transfers_in=_rows_for_ids(names, in_ids),
        legal=True,
        note=note,
    )


def _pool_view(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["element_id"] = pd.to_numeric(out["element_id"], errors="coerce")
    out = out.dropna(subset=["element_id"])
    out["element_id"] = out["element_id"].astype(int)
    out["position"] = out["position"].map(normalize_position)
    out["cost_m"] = pd.to_numeric(out["cost_m"], errors="coerce").fillna(0.0)
    out["xpts"] = pd.to_numeric(out["xpts"], errors="coerce").fillna(0.0)
    out["club"] = club_key(out).astype(str)
    return out.drop_duplicates("element_id")


def _illegal_option(
    current: set[int],
    new_ids: set[int],
    names: pd.DataFrame,
    *,
    n: int,
    hits: int,
    key: str,
    label: str,
    note: str,
) -> dict[str, Any]:
    return _option_dict(
        key=key,
        label=label,
        n_transfers=n,
        hits=hits,
        expected_net=0.0,
        transfers_out=_rows_for_ids(names, current - new_ids),
        transfers_in=_rows_for_ids(names, new_ids - current),
        legal=False,
        note=note or "This move does not fit — budget or three-per-club. You cannot take it on its own.",
    )


def _score_locked_squad(
    work: pd.DataFrame,
    current: set[int],
    new_ids: set[int],
    hold: SquadSolution,
    *,
    budget: float,
    names: pd.DataFrame,
    free_transfers: int,
    hit_cost: float,
    captain_min_p_play: float,
    season: str | None,
    event: int | None,
    key: str,
    label: str,
    note: str = "",
) -> dict[str, Any] | None:
    if len(new_ids) != SQUAD_SIZE:
        return None
    n = len(current - new_ids)
    if n <= 0:
        return None
    hits = max(0, n - int(free_transfers))
    eids = pd.to_numeric(work["element_id"], errors="coerce")
    slim = work.loc[eids.isin(new_ids)].drop_duplicates("element_id")
    have = set(pd.to_numeric(slim["element_id"], errors="coerce").dropna().astype(int))
    if have != set(new_ids) or legality_errors(slim, budget_m=budget):
        return _illegal_option(
            current, new_ids, names, n=n, hits=hits, key=key, label=label, note=note
        )
    try:
        sol = solve_squad(
            slim,
            budget_m=budget,
            captain_min_p_play=captain_min_p_play,
            season=season,
            event=event,
            locked_ids=new_ids,
            time_limit=LOCKED_XI_TIME,
        )
    except RuntimeError:
        return _illegal_option(
            current, new_ids, names, n=n, hits=hits, key=key, label=label, note=note
        )
    net = float(sol.objective - hits * float(hit_cost) - hold.objective)
    out_ids = current - sol.squad_ids
    in_ids = sol.squad_ids - current
    return _option_dict(
        key=key,
        label=label,
        n_transfers=n,
        hits=hits,
        expected_net=net,
        transfers_out=_rows_for_ids(names, out_ids),
        transfers_in=_rows_for_ids(names, in_ids),
        legal=True,
        note=note,
    )


def _enumerate_singles(
    view: pd.DataFrame,
    current: set[int],
    budget: float,
) -> list[tuple[int, int, float]]:
    """Legal same-position 1-for-1 swaps, ranked by incoming minus outgoing xPts."""
    cur = view.loc[view["element_id"].isin(current)]
    mkt = view.loc[~view["element_id"].isin(current)]
    if cur.empty or mkt.empty:
        return []
    clubs = Counter(cur["club"].astype(str))
    spent = float(cur["cost_m"].sum())
    found: list[tuple[int, int, float]] = []
    for pos in POSITION_ORDER:
        owned = cur.loc[cur["position"] == pos]
        market = mkt.loc[mkt["position"] == pos].nlargest(MENU_MARKET_PER_POS, "xpts")
        if owned.empty or market.empty:
            continue
        for _, out in owned.iterrows():
            out_id = int(out["element_id"])
            out_x = float(out["xpts"])
            out_cost = float(out["cost_m"])
            out_club = str(out["club"])
            for _, inn in market.iterrows():
                in_x = float(inn["xpts"])
                if in_x < out_x - 0.5:
                    continue
                in_club = str(inn["club"])
                in_count = clubs[in_club] - (1 if out_club == in_club else 0)
                if in_count + 1 > MAX_PER_CLUB:
                    continue
                if spent - out_cost + float(inn["cost_m"]) > budget + 1e-9:
                    continue
                found.append((out_id, int(inn["element_id"]), in_x - out_x))
    found.sort(key=lambda row: -row[2])
    return found


def _pick_diverse_singles(
    cands: list[tuple[int, int, float]],
    *,
    limit: int,
) -> list[tuple[int, int, float]]:
    """Prefer different incoming players, then different sales, then a second target."""
    picked: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    def _take(*, in_cap: int, out_cap: int) -> None:
        in_n: Counter[int] = Counter()
        out_n: Counter[int] = Counter()
        for out_id, in_id, _crude in picked:
            in_n[in_id] += 1
            out_n[out_id] += 1
        for row in cands:
            if len(picked) >= limit:
                return
            out_id, in_id, _crude = row
            fp = (out_id, in_id)
            if fp in seen:
                continue
            if in_n[in_id] >= in_cap or out_n[out_id] >= out_cap:
                continue
            seen.add(fp)
            picked.append(row)
            in_n[in_id] += 1
            out_n[out_id] += 1

    _take(in_cap=1, out_cap=1)
    _take(in_cap=1, out_cap=2)
    _take(in_cap=2, out_cap=2)
    return picked


def _move_label(opt: dict[str, Any]) -> str:
    left = ", ".join(str(r.get("name") or "?") for r in opt.get("transfers_out") or []) or "?"
    right = ", ".join(str(r.get("name") or "?") for r in opt.get("transfers_in") or []) or "?"
    return f"{left} → {right}"


def build_transfer_menu(
    work: pd.DataFrame,
    current: set[int],
    hold: SquadSolution,
    chosen: SquadSolution,
    names: pd.DataFrame,
    *,
    budget: float,
    free_transfers: int,
    max_transfers: int,
    hit_cost: float,
    captain_min_p_play: float,
    season: str | None,
    event: int | None,
) -> list[dict[str, Any]]:
    """Other ideas besides the headline plan. Enumerated 1-for-1s, not extra full-market ILPs.

    A two-move TAKE is a bundle. If the user only likes one half, or the bank is tight,
    they need several legal singles and each half scored on its own.
    """
    headline = _option_fingerprint(current - chosen.squad_ids, chosen.squad_ids - current)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = {headline}
    menu: list[dict[str, Any]] = []
    view = _pool_view(work)
    score_kw: dict[str, Any] = {
        "budget": budget,
        "names": names,
        "free_transfers": free_transfers,
        "hit_cost": hit_cost,
        "captain_min_p_play": captain_min_p_play,
        "season": season,
        "event": event,
    }

    def _add(opt: dict[str, Any] | None) -> None:
        if not opt:
            return
        fp = _option_fingerprint(_id_set(opt["transfers_out"]), _id_set(opt["transfers_in"]))
        if not fp[0]:
            return
        if fp in seen:
            if str(opt.get("key") or "").startswith("only_"):
                for row in menu:
                    existing = _option_fingerprint(
                        _id_set(row["transfers_out"]), _id_set(row["transfers_in"])
                    )
                    if existing != fp:
                        continue
                    row["label"] = opt.get("label") or row.get("label")
                    extra = " This is one half of the headline package, scored on its own."
                    note = row.get("note") or ""
                    if "half of the headline" not in note:
                        row["note"] = (note + extra).strip()
                    if not opt.get("legal", True):
                        row["legal"] = False
                        row["note"] = opt.get("note") or row["note"]
                    break
            return
        seen.add(fp)
        menu.append(opt)

    if int(chosen.n_transfers) >= 2:
        outs = _rows_for_ids(names, current - chosen.squad_ids)
        ins = _rows_for_ids(names, chosen.squad_ids - current)
        for i, (left, right) in enumerate(_pair_by_position(outs, ins)):
            try:
                out_id = int(left["element_id"])
                in_id = int(right["element_id"])
            except (KeyError, TypeError, ValueError):
                continue
            out_name = left.get("name") or "out"
            in_name = right.get("name") or "in"
            _add(
                _score_locked_squad(
                    work,
                    current,
                    (current - {out_id}) | {in_id},
                    hold,
                    key=f"only_{i}",
                    label=f"Only {out_name} → {in_name}",
                    note=(
                        "This is one half of the headline package, scored as if you ignore the other move. "
                        "Use it if you like this idea and not the rest."
                    ),
                    **score_kw,
                )
            )

    scored_singles: list[dict[str, Any]] = []
    for out_id, in_id, _crude in _pick_diverse_singles(
        _enumerate_singles(view, current, budget),
        limit=MENU_SINGLE_SCORE,
    ):
        opt = _score_locked_squad(
            work,
            current,
            (current - {out_id}) | {in_id},
            hold,
            key="single",
            label="Single",
            note="One move. Take this instead of the headline if it is the swap you actually want.",
            **score_kw,
        )
        if opt and opt.get("legal"):
            scored_singles.append(opt)
    scored_singles.sort(key=lambda row: -float(row["expected_net"]))
    headline_is_single = int(chosen.n_transfers) == 1
    headline_ins = chosen.squad_ids - current
    kept = 0
    named_best = False
    named_runner = False
    used_ins: set[int] = set(headline_ins) if headline_is_single else set()
    for opt in scored_singles:
        if kept >= MENU_SINGLE_KEEP:
            break
        ins = _id_set(opt["transfers_in"])
        fresh_in = bool(ins) and ins.isdisjoint(used_ins)
        if not headline_is_single and not named_best:
            opt["key"] = "best_1"
            opt["label"] = "Best single transfer"
            opt["note"] = (
                "Do this if you only want one move — free transfer, no need to like the rest of the package."
            )
        elif not named_runner and fresh_in:
            opt["key"] = "runner_up_1"
            opt["label"] = "Next-best single transfer"
            opt["note"] = "If you do not like the best single, this is the next legal one-move."
        else:
            opt["key"] = f"single_{kept}"
            opt["label"] = _move_label(opt)
        before = len(menu)
        _add(opt)
        if len(menu) <= before:
            continue
        if opt["key"] == "best_1":
            named_best = True
        elif opt["key"] == "runner_up_1":
            named_runner = True
        used_ins |= ins
        kept += 1

    if int(max_transfers) >= 2:
        ones = [row for row in menu if row.get("legal") and int(row.get("n_transfers") or 0) == 1][:8]
        pair_cands: list[tuple[set[int], dict[str, Any], dict[str, Any]]] = []
        for i, left in enumerate(ones):
            left_out = _id_set(left["transfers_out"])
            left_in = _id_set(left["transfers_in"])
            for right in ones[i + 1 :]:
                right_out = _id_set(right["transfers_out"])
                right_in = _id_set(right["transfers_in"])
                if left_out & right_out or left_in & right_in:
                    continue
                new_ids = (current - left_out - right_out) | left_in | right_in
                if len(new_ids) != SQUAD_SIZE:
                    continue
                pair_cands.append((new_ids, left, right))
        pair_cands.sort(
            key=lambda item: -(
                float(item[1].get("expected_net") or 0) + float(item[2].get("expected_net") or 0)
            )
        )
        pairs_kept = 0
        for i, (new_ids, left, right) in enumerate(pair_cands[:MENU_PAIR_SCORE]):
            if pairs_kept >= MENU_PAIR_KEEP:
                break
            opt = _score_locked_squad(
                work,
                current,
                new_ids,
                hold,
                key=f"pair_{i}",
                label=f"Two moves: {_move_label(left)}; {_move_label(right)}",
                note="A two-move package you can take instead of the headline. Still a bundle.",
                **score_kw,
            )
            if not opt or not opt.get("legal") or int(opt.get("n_transfers") or 0) != 2:
                continue
            before = len(menu)
            _add(opt)
            if len(menu) > before:
                pairs_kept += 1

    legal = [row for row in menu if row.get("legal")]
    illegal = [row for row in menu if not row.get("legal")]
    legal.sort(key=lambda row: -float(row["expected_net"]))
    return legal + illegal


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
    alternatives: list[dict[str, Any]] = []
    if mode == "myopic":
        alternatives = build_transfer_menu(
            work,
            current,
            hold,
            chosen,
            names,
            budget=budget,
            free_transfers=int(free_transfers),
            max_transfers=int(max_transfers),
            hit_cost=float(hit_cost),
            captain_min_p_play=captain_min_p_play,
            season=season,
            event=event,
        )
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
        alternatives=alternatives,
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
    if plan.alternatives:
        lines.append("Other legal ideas (you can take one of these instead of the headline package)")
        for alt in plan.alternatives:
            left = ", ".join(str(r.get("name")) for r in alt.get("transfers_out") or []) or "-"
            right = ", ".join(str(r.get("name")) for r in alt.get("transfers_in") or []) or "-"
            legal = "" if alt.get("legal", True) else "  ILLEGAL on its own"
            lines.append(
                f"  {alt.get('label')}: {left} -> {right}  vs hold {alt.get('expected_net', 0):+.2f}"
                f"{legal}"
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
        "alternatives": plan.alternatives,
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
    squad_ids: list[int] | set[int] | None = None,
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
    loaded = load_prediction_pool(
        panel,
        settings=cfg,
        season=season,
        event=event,
        overlay_live=overlay_live,
    )
    print(f"Scoring {loaded.season} GW{loaded.event} ({loaded.source}) for transfers...", flush=True)
    pool = loaded.pool
    season = loaded.season
    event = loaded.event

    current_meta = None
    if squad_ids is not None:
        current_ids = {int(i) for i in squad_ids}
        if len(current_ids) != SQUAD_SIZE:
            raise RuntimeError(f"squad_ids has {len(current_ids)} unique ids; need {SQUAD_SIZE}")
    elif squad_path is not None:
        current_ids = set(load_squad_csv(Path(squad_path)))
    elif team_id is not None:
        picks, api_bank = fetch_entry_picks(int(team_id), int(event), settings=cfg)
        current_ids = set(picks)
        if bank_m is None:
            bank_m = api_bank
    elif loaded.source == "live_prior":
        raise RuntimeError(
            "Live GW has no previous solver squad. Pass --squad data/overrides/squad.csv "
            "or --team-id (your FPL entry id)."
        )
    else:
        scored = score_season(panel, str(season))
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
        "note": loaded.note,
        "pool_source": loaded.source,
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
