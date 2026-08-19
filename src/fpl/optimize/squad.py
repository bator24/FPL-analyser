"""Single-GW PuLP squad, XI, and captain solver."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd
import pulp

from fpl.config import Settings, load_settings
from fpl.optimize.pool import load_prediction_pool, with_risk_columns
from fpl.optimize.rules import (
    BUDGET_M,
    CAPTAIN_MIN_P_PLAY,
    HIT_COST,
    MAX_PER_CLUB,
    MAX_TRANSFERS_DEFAULT,
    POSITION_ORDER,
    SQUAD_COUNTS,
    SQUAD_SIZE,
    XI_BOUNDS,
    XI_SIZE,
    club_key,
    formation_from_xi,
    is_legal,
    legality_errors,
    normalize_position,
    xi_shape_options,
)


@dataclass
class SquadSolution:
    table: pd.DataFrame
    objective: float
    naive_objective: float
    budget_m: float
    bank_m: float
    formation: str
    season: str | None
    event: int | None
    captain_min_p_play: float
    solver_status: str
    n_transfers: int = 0
    hits: int = 0
    hit_cost: float = 0.0

    @property
    def haircut(self) -> float:
        return float(self.naive_objective - self.objective)

    @property
    def net_objective(self) -> float:
        return float(self.objective - self.hits * self.hit_cost)

    @property
    def squad_ids(self) -> set[int]:
        return set(pd.to_numeric(self.table["element_id"], errors="coerce").astype(int))

    @property
    def captain_id(self) -> int:
        caps = self.table.loc[self.table["is_captain"], "element_id"]
        return int(caps.iloc[0])

    @property
    def vice_id(self) -> int:
        vices = self.table.loc[self.table["is_vice"], "element_id"]
        return int(vices.iloc[0])

    @property
    def xi(self) -> pd.DataFrame:
        return self.table.loc[self.table["in_xi"]].copy()

    @property
    def legal(self) -> bool:
        return is_legal(
            self.table,
            xi=self.xi,
            captain_id=self.captain_id,
            vice_id=self.vice_id,
            budget_m=self.budget_m,
            captain_min_p_play=self.captain_min_p_play,
        )


def _prepare(pool: pd.DataFrame) -> pd.DataFrame:
    out = with_risk_columns(pool.copy())
    out["element_id"] = pd.to_numeric(out["element_id"], errors="coerce")
    out = out.dropna(subset=["element_id"])
    out["element_id"] = out["element_id"].astype(int)
    out["position"] = out["position"].map(normalize_position)
    out["cost_m"] = pd.to_numeric(out["cost_m"], errors="coerce").fillna(0)
    out["cost_tenths"] = (out["cost_m"] * 10.0).round().astype(int)
    out["club"] = club_key(out)
    if "name" not in out.columns:
        out["name"] = out["element_id"].astype(str)
    out["name"] = out["name"].astype("string").fillna(out["element_id"].astype(str))
    if out["element_id"].duplicated().any():
        raise ValueError("Pool must be unique on element_id (collapse DGWs first)")
    return out.reset_index(drop=True)


def _pick_captain(xi: pd.DataFrame, captain_min_p_play: float) -> pd.Series:
    eligible = xi[xi["p_play"] >= captain_min_p_play]
    pool = eligible if len(eligible) else xi
    # Among those who clear the minutes gate, double unconditional EV — do not
    # inflate 75–90% players by dividing by p_play.
    return pool.sort_values(["xpts", "p_play", "xpts_if_plays"], ascending=False).iloc[0]


def _pick_vice(xi: pd.DataFrame, captain_id: int, captain_min_p_play: float) -> pd.Series:
    rest = xi[xi["element_id"] != captain_id]
    eligible = rest[rest["p_play"] >= captain_min_p_play]
    pool = eligible if len(eligible) else rest
    return pool.sort_values(["xpts", "p_play", "xpts_if_plays"], ascending=False).iloc[0]


def _objective(xi: pd.DataFrame, captain: pd.Series) -> tuple[float, float]:
    risk = float(xi["xpts"].sum() + captain["xpts"])
    naive = float(xi["xpts_if_plays"].sum() + captain["xpts_if_plays"])
    return risk, naive


def _assemble(
    squad: pd.DataFrame,
    xi_ids: set[int],
    captain_id: int,
    vice_id: int,
    *,
    budget_m: float,
    season: str | None,
    event: int | None,
    captain_min_p_play: float,
    solver_status: str,
    n_transfers: int = 0,
    hits: int = 0,
    hit_cost: float = 0.0,
) -> SquadSolution:
    table = squad.copy()
    table["in_xi"] = table["element_id"].isin(xi_ids)
    table["is_captain"] = table["element_id"] == captain_id
    table["is_vice"] = table["element_id"] == vice_id
    bench = table.loc[~table["in_xi"]].copy()
    bench_gk = bench[bench["position"] == "GKP"].sort_values("xpts", ascending=False)
    bench_out = bench[bench["position"] != "GKP"].sort_values(
        ["xpts", "p_play"], ascending=False
    )
    ordered_bench = pd.concat([bench_out, bench_gk], ignore_index=True)
    bench_order = {int(eid): i + 1 for i, eid in enumerate(ordered_bench["element_id"])}
    table["bench_order"] = table["element_id"].map(lambda eid: bench_order.get(int(eid), 0))
    xi = table.loc[table["in_xi"]]
    captain = table.loc[table["is_captain"]].iloc[0]
    risk, naive = _objective(xi, captain)
    spent = float(table["cost_m"].sum())
    bank = float(budget_m - spent)
    if abs(bank) < 1e-8:
        bank = 0.0
    return SquadSolution(
        table=_order_table(table),
        objective=risk,
        naive_objective=naive,
        budget_m=budget_m,
        bank_m=bank,
        formation=formation_from_xi(xi),
        season=season,
        event=event,
        captain_min_p_play=captain_min_p_play,
        solver_status=solver_status,
        n_transfers=n_transfers,
        hits=hits,
        hit_cost=hit_cost,
    )


def _order_table(table: pd.DataFrame) -> pd.DataFrame:
    pos_rank = {pos: i for i, pos in enumerate(POSITION_ORDER)}
    out = table.copy()
    out["_pos"] = out["position"].map(pos_rank)
    out["_xi"] = (~out["in_xi"]).astype(int)
    out = out.sort_values(["_xi", "_pos", "xpts"], ascending=[True, True, False])
    return out.drop(columns=["_pos", "_xi"]).reset_index(drop=True)


def solve_squad(
    pool: pd.DataFrame,
    *,
    budget_m: float = BUDGET_M,
    max_per_club: int = MAX_PER_CLUB,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
    season: str | None = None,
    event: int | None = None,
    time_limit: int = 60,
    locked_ids: set[int] | None = None,
    current_ids: set[int] | None = None,
    max_transfers: int | None = None,
    free_transfers: int = 1,
    hit_cost: float = HIT_COST,
) -> SquadSolution:
    """Risk-adjusted 15 + XI + captain. Captain is 2x unconditional xPts, gated on p_play.

    locked_ids forces this exact 15 (hold / XI-only).
    current_ids transfers from this 15; objective subtracts hit_cost per hit.
    """
    players = _prepare(pool)
    ids = players["element_id"].tolist()
    by_id = players.set_index("element_id")
    budget_tenths = int(round(budget_m * 10))
    id_set = set(ids)

    locked: set[int] | None = None
    if locked_ids is not None:
        locked = {int(i) for i in locked_ids}
        missing = locked - id_set
        if missing:
            raise RuntimeError(f"Locked squad players missing from pool: {sorted(missing)}")
        if len(locked) != SQUAD_SIZE:
            raise RuntimeError(f"Locked squad size {len(locked)} != {SQUAD_SIZE}")

    current: set[int] | None = None
    transfer_cap = 0
    if locked is None and current_ids is not None:
        current = {int(i) for i in current_ids}
        missing = current - id_set
        if missing:
            raise RuntimeError(f"Current squad players missing from pool: {sorted(missing)}")
        if len(current) != SQUAD_SIZE:
            raise RuntimeError(f"Current squad size {len(current)} != {SQUAD_SIZE}")
        transfer_cap = MAX_TRANSFERS_DEFAULT if max_transfers is None else int(max_transfers)

    prob = pulp.LpProblem("fpl_single_gw", pulp.LpMaximize)
    squad = prob.add_variable_dicts("squad", ids, cat="Binary")
    xi = prob.add_variable_dicts("xi", ids, cat="Binary")
    cap = prob.add_variable_dicts("cap", ids, cat="Binary")
    hits_var = None

    for eid in ids:
        prob += xi[eid] <= squad[eid]
        prob += cap[eid] <= xi[eid]
    if locked is not None:
        for eid in ids:
            prob += squad[eid] == (1 if eid in locked else 0)
    else:
        prob += pulp.lpSum(squad[eid] for eid in ids) == SQUAD_SIZE
    prob += pulp.lpSum(xi[eid] for eid in ids) == XI_SIZE
    prob += pulp.lpSum(cap[eid] for eid in ids) == 1
    for pos, need in SQUAD_COUNTS.items():
        pos_ids = [eid for eid in ids if by_id.at[eid, "position"] == pos]
        lo, hi = XI_BOUNDS[pos]
        if locked is None:
            prob += pulp.lpSum(squad[eid] for eid in pos_ids) == need
        prob += pulp.lpSum(xi[eid] for eid in pos_ids) >= lo
        prob += pulp.lpSum(xi[eid] for eid in pos_ids) <= hi
    prob += pulp.lpSum(squad[eid] * int(by_id.at[eid, "cost_tenths"]) for eid in ids) <= budget_tenths
    for club, group in players.groupby("club"):
        club_ids = group["element_id"].tolist()
        if club_ids:
            prob += pulp.lpSum(squad[eid] for eid in club_ids) <= max_per_club

    if current is not None:
        n_kept = pulp.lpSum(squad[eid] for eid in current)
        prob += n_kept >= SQUAD_SIZE - transfer_cap
        hits_var = pulp.LpVariable("hits", lowBound=0, upBound=max(transfer_cap, 0), cat="Integer")
        prob += hits_var >= (SQUAD_SIZE - n_kept) - int(free_transfers)

    eligible_ids = [eid for eid in ids if float(by_id.at[eid, "p_play"]) >= captain_min_p_play]
    if locked is not None:
        eligible_ids = [eid for eid in eligible_ids if eid in locked]
    if eligible_ids:
        for eid in ids:
            if eid not in eligible_ids:
                prob += cap[eid] == 0
        prob += pulp.lpSum(xi[eid] for eid in eligible_ids) >= 1

    tie = {eid: (int(eid) % 1000) * 1e-9 for eid in ids}
    obj = pulp.lpSum(
        xi[eid] * float(by_id.at[eid, "xpts"])
        + cap[eid] * float(by_id.at[eid, "xpts"])
        + squad[eid] * tie[eid]
        for eid in ids
    )
    if hits_var is not None:
        obj -= float(hit_cost) * hits_var
    prob += obj

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status_code = prob.solve(solver)
    status = pulp.LpStatus.get(status_code, str(status_code))
    if status != "Optimal":
        raise RuntimeError(f"Squad ILP failed ({status}). Check pool size, budget, and positions.")

    squad_ids = [eid for eid in ids if pulp.value(squad[eid]) > 0.5]
    xi_ids = {eid for eid in ids if pulp.value(xi[eid]) > 0.5}
    cap_ids = [eid for eid in ids if pulp.value(cap[eid]) > 0.5]
    if len(cap_ids) != 1:
        raise RuntimeError("Solver did not pick a unique captain")
    picked = players[players["element_id"].isin(squad_ids)].copy()
    xi_rows = picked[picked["element_id"].isin(xi_ids)]
    vice = _pick_vice(xi_rows, cap_ids[0], captain_min_p_play)
    if current is not None:
        n_transfers = SQUAD_SIZE - len(current.intersection(squad_ids))
        hits = int(round(float(pulp.value(hits_var) or 0))) if hits_var is not None else 0
        applied_hit_cost = float(hit_cost)
    else:
        n_transfers = 0
        hits = 0
        applied_hit_cost = 0.0
    solution = _assemble(
        picked,
        xi_ids,
        cap_ids[0],
        int(vice["element_id"]),
        budget_m=budget_m,
        season=season,
        event=event,
        captain_min_p_play=captain_min_p_play,
        solver_status=status,
        n_transfers=n_transfers,
        hits=hits,
        hit_cost=applied_hit_cost,
    )
    errors = legality_errors(
        solution.table,
        xi=solution.xi,
        captain_id=solution.captain_id,
        vice_id=solution.vice_id,
        budget_m=budget_m,
        max_per_club=max_per_club,
        captain_min_p_play=captain_min_p_play,
    )
    if errors:
        raise RuntimeError("Illegal squad from solver: " + "; ".join(errors))
    return solution



def brute_force_squad(
    pool: pd.DataFrame,
    *,
    budget_m: float = BUDGET_M,
    max_per_club: int = MAX_PER_CLUB,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
    season: str | None = None,
    event: int | None = None,
) -> SquadSolution:
    """Enumerate legal 15s + XIs. For toy pools only (n around 15–20)."""
    players = _prepare(pool)
    if len(players) > 22:
        raise ValueError("brute_force_squad is only for tiny toy pools")
    by_pos = {pos: players[players["position"] == pos] for pos in POSITION_ORDER}
    for pos, need in SQUAD_COUNTS.items():
        if len(by_pos[pos]) < need:
            raise RuntimeError(f"Not enough {pos} players for a legal squad")

    best: tuple[float, set[int], set[int], int] | None = None
    gkp_opts = list(combinations(by_pos["GKP"]["element_id"].tolist(), SQUAD_COUNTS["GKP"]))
    def_opts = list(combinations(by_pos["DEF"]["element_id"].tolist(), SQUAD_COUNTS["DEF"]))
    mid_opts = list(combinations(by_pos["MID"]["element_id"].tolist(), SQUAD_COUNTS["MID"]))
    fwd_opts = list(combinations(by_pos["FWD"]["element_id"].tolist(), SQUAD_COUNTS["FWD"]))
    budget_tenths = int(round(budget_m * 10))
    indexed = players.set_index("element_id")

    for gkp in gkp_opts:
        for defs in def_opts:
            for mids in mid_opts:
                for fwds in fwd_opts:
                    squad_ids = set(gkp + defs + mids + fwds)
                    spent = int(indexed.loc[list(squad_ids), "cost_tenths"].sum())
                    if spent > budget_tenths:
                        continue
                    clubs = indexed.loc[list(squad_ids), "club"].value_counts()
                    if (clubs > max_per_club).any():
                        continue
                    squad_rows = indexed.loc[list(squad_ids)]
                    gkp_ids = [i for i in gkp]
                    def_ids = list(defs)
                    mid_ids = list(mids)
                    fwd_ids = list(fwds)
                    for n_def, n_mid, n_fwd in xi_shape_options():
                        for xi_g in gkp_ids:
                            for xi_d in combinations(def_ids, n_def):
                                for xi_m in combinations(mid_ids, n_mid):
                                    for xi_f in combinations(fwd_ids, n_fwd):
                                        xi_ids = {xi_g, *xi_d, *xi_m, *xi_f}
                                        xi_rows = squad_rows.loc[list(xi_ids)].reset_index()
                                        captain = _pick_captain(xi_rows, captain_min_p_play)
                                        risk, _naive = _objective(xi_rows, captain)
                                        key = (risk, xi_ids, squad_ids, int(captain["element_id"]))
                                        if best is None or risk > best[0] + 1e-12:
                                            best = key
    if best is None:
        raise RuntimeError("No legal squad in brute-force pool")
    _risk, xi_ids, squad_ids, captain_id = best
    picked = players[players["element_id"].isin(squad_ids)].copy()
    xi_rows = picked[picked["element_id"].isin(xi_ids)]
    vice = _pick_vice(xi_rows, captain_id, captain_min_p_play)
    return _assemble(
        picked,
        xi_ids,
        captain_id,
        int(vice["element_id"]),
        budget_m=budget_m,
        season=season,
        event=event,
        captain_min_p_play=captain_min_p_play,
        solver_status="BruteForce",
    )


def solution_to_dict(solution: SquadSolution) -> dict[str, Any]:
    table = solution.table
    keep = [
        c
        for c in [
            "element_id",
            "name",
            "position",
            "team",
            "team_id",
            "cost_m",
            "xpts",
            "xpts_if_plays",
            "p_play",
            "in_xi",
            "is_captain",
            "is_vice",
            "bench_order",
        ]
        if c in table.columns
    ]
    payload = {
        "season": solution.season,
        "event": solution.event,
        "budget_m": solution.budget_m,
        "bank_m": round(solution.bank_m, 1),
        "formation": solution.formation,
        "objective": round(solution.objective, 4),
        "naive_objective": round(solution.naive_objective, 4),
        "haircut": round(solution.haircut, 4),
        "captain_min_p_play": solution.captain_min_p_play,
        "legal": solution.legal,
        "solver_status": solution.solver_status,
        "captain_id": solution.captain_id,
        "vice_id": solution.vice_id,
        "n_transfers": solution.n_transfers,
        "hits": solution.hits,
        "hit_cost": solution.hit_cost,
        "net_objective": round(solution.net_objective, 4),
        "players": table[keep].to_dict(orient="records"),
    }
    return payload


def format_squad_report(solution: SquadSolution) -> str:
    table = solution.table
    season = solution.season or "?"
    event = solution.event if solution.event is not None else "?"
    lines = [
        "Single-GW ILP squad",
        f"Season {season}  event {event}",
        f"Budget {solution.budget_m:.1f}  spent {solution.budget_m - solution.bank_m:.1f}  bank {solution.bank_m:.1f}",
        f"Formation {solution.formation}",
        (
            f"Objective {solution.objective:.2f} risk-adjusted  |  "
            f"naive-if-plays {solution.naive_objective:.2f}  haircut {solution.haircut:.2f}"
        ),
        f"Captain gate: p_play >= {solution.captain_min_p_play:.2f}",
        "",
        "XI",
    ]
    xi = table.loc[table["in_xi"]]
    for _, row in xi.iterrows():
        flag = ""
        if row["is_captain"]:
            flag = " C"
        elif row["is_vice"]:
            flag = " V"
        lines.append(
            f"  {row['position']:<3} {str(row['name']):<22} {row['cost_m']:>5.1f}  "
            f"xpts {row['xpts']:5.2f}  if-plays {row['xpts_if_plays']:5.2f}  "
            f"p_play {row['p_play']:.2f}{flag}"
        )
    lines.append("\nBench (order; auto-subs later)")
    bench = table.loc[~table["in_xi"]].sort_values("bench_order")
    for _, row in bench.iterrows():
        lines.append(
            f"  {int(row['bench_order'])} {row['position']:<3} {str(row['name']):<22} {row['cost_m']:>5.1f}  "
            f"xpts {row['xpts']:5.2f}  p_play {row['p_play']:.2f}"
        )
    lines.extend(
        [
            "",
            f"Legal: {solution.legal}",
            "Objective is XI unconditional xPts + captain double (still unconditional). "
            "Naive-if-plays treats every starter as nailed. Captain eligibility is p_play-gated.",
        ]
    )
    return "\n".join(lines)


def run_squad(
    *,
    settings: Settings | None = None,
    season: str | None = None,
    event: int | None = None,
    budget_m: float = BUDGET_M,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
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
    print(f"Scoring {loaded.season} GW{loaded.event} ({loaded.source}) and solving ILP...", flush=True)
    pool = loaded.pool
    used_live = loaded.source == "live_prior" or overlay_live
    solution = solve_squad(
        pool,
        budget_m=budget_m,
        captain_min_p_play=captain_min_p_play,
        season=loaded.season,
        event=loaded.event,
    )
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        c
        for c in [
            "element_id",
            "name",
            "position",
            "team",
            "team_id",
            "cost_m",
            "xpts",
            "xpts_if_plays",
            "p_play",
            "in_xi",
            "is_captain",
            "is_vice",
            "bench_order",
        ]
        if c in solution.table.columns
    ]
    out_path = cfg.processed_dir / "squad.parquet"
    eval_path = cfg.eval_dir / "squad.json"
    solution.table[keep].to_parquet(out_path, index=False)
    payload = solution_to_dict(solution)
    payload["n_pool"] = int(len(pool))
    payload["overlay_live_prices"] = used_live
    payload["pool_source"] = loaded.source
    payload["n_mapped"] = loaded.n_mapped
    payload["n_unmapped"] = loaded.n_unmapped
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "solution": solution,
        "pool": pool,
        "eval_path": eval_path,
        "squad_path": out_path,
        "note": loaded.note,
        "pool_source": loaded.source,
    }
