"""This-GW chip EV. One chip a week; no season option value in v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpl.config import Settings, load_settings
from fpl.optimize.rules import CAPTAIN_MIN_P_PLAY, FREE_TRANSFERS_DEFAULT, MAX_TRANSFERS_DEFAULT
from fpl.optimize.squad import SquadSolution
from fpl.optimize.transfers import TransferPlan, run_transfer


@dataclass
class ChipOption:
    key: str
    label: str
    ev: float
    gain_vs_hold: float
    is_chip: bool
    note: str


@dataclass
class ChipReport:
    season: str | None
    event: int | None
    options: list[ChipOption]
    best_this_gw: str
    best_no_chip: str
    best_chip: str | None
    chip_beats_no_chip: bool

    def option(self, key: str) -> ChipOption:
        for row in self.options:
            if row.key == key:
                return row
        raise KeyError(key)


def _captain_xpts(sol: SquadSolution) -> tuple[float, str]:
    row = sol.table.loc[sol.table["is_captain"]].iloc[0]
    return float(row["xpts"]), str(row["name"])


def bench_xpts(sol: SquadSolution) -> float:
    return float(sol.table.loc[~sol.table["in_xi"], "xpts"].sum())


def bb_ev(sol: SquadSolution) -> float:
    """All 15 score; captain still doubled. Gain vs that squad's XI is bench xPts."""
    return float(sol.objective + bench_xpts(sol))


def tc_ev(sol: SquadSolution) -> float:
    """Captain trebled. Gain vs that squad's 2× captain is one extra copy of captain xPts."""
    extra, _name = _captain_xpts(sol)
    return float(sol.objective + extra)


def _rebuild(plan: TransferPlan) -> SquadSolution | None:
    if plan.wildcard is not None:
        return plan.wildcard
    if plan.mode == "wildcard":
        return plan.chosen
    return None


def evaluate_chips(plan: TransferPlan) -> ChipReport:
    hold = plan.hold
    fielded = plan.chosen
    hold_ev = float(hold.objective)
    cap_hold, cap_hold_name = _captain_xpts(hold)
    cap_field, cap_field_name = _captain_xpts(fielded)
    bench_hold = bench_xpts(hold)
    bench_field = bench_xpts(fielded)
    transfer_net = float(fielded.net_objective)
    same_squad = fielded.squad_ids == hold.squad_ids

    options = [
        ChipOption("hold", "Hold (no chip)", hold_ev, 0.0, False, "Re-pick XI and captain only."),
        ChipOption(
            "transfers",
            "Transfers (no chip)",
            transfer_net,
            transfer_net - hold_ev,
            False,
            f"{plan.n_transfers} moves, {plan.hits} hits."
            if not same_squad
            else "No moves vs hold.",
        ),
        ChipOption(
            "bb_hold",
            "Bench Boost on hold 15",
            bb_ev(hold),
            bb_ev(hold) - hold_ev,
            True,
            f"Bench xPts {bench_hold:.2f}. Cannot combine with WC/FH/TC.",
        ),
        ChipOption(
            "bb_transfers",
            "Bench Boost after transfers",
            bb_ev(fielded) - fielded.hits * fielded.hit_cost,
            (bb_ev(fielded) - fielded.hits * fielded.hit_cost) - hold_ev,
            True,
            f"Bench xPts {bench_field:.2f} after hits. Enabler benches make BB weaker.",
        ),
        ChipOption(
            "tc_hold",
            f"Triple Captain ({cap_hold_name}) on hold",
            tc_ev(hold),
            tc_ev(hold) - hold_ev,
            True,
            f"Extra copy of {cap_hold_name} ({cap_hold:.2f}). Cannot combine with BB/WC/FH.",
        ),
        ChipOption(
            "tc_transfers",
            f"Triple Captain ({cap_field_name}) after transfers",
            tc_ev(fielded) - fielded.hits * fielded.hit_cost,
            (tc_ev(fielded) - fielded.hits * fielded.hit_cost) - hold_ev,
            True,
            f"Extra copy of {cap_field_name} ({cap_field:.2f}) after hits.",
        ),
    ]
    rebuild = _rebuild(plan)
    if rebuild is not None:
        options.append(
            ChipOption(
                "fh_wc",
                "Free Hit / Wildcard rebuild",
                float(rebuild.objective),
                float(rebuild.objective) - hold_ev,
                True,
                "Same this-GW EV. FH reverts next week; WC keeps the 15. Not both with BB/TC.",
            )
        )

    best = max(options, key=lambda row: (row.ev, -int(row.is_chip)))
    no_chip = max((row for row in options if not row.is_chip), key=lambda row: row.ev)
    chips = [row for row in options if row.is_chip]
    best_chip = max(chips, key=lambda row: row.ev) if chips else None
    chip_beats = bool(best_chip and best_chip.ev > no_chip.ev + 0.05)

    return ChipReport(
        season=hold.season,
        event=hold.event,
        options=options,
        best_this_gw=best.key,
        best_no_chip=no_chip.key,
        best_chip=best_chip.key if best_chip else None,
        chip_beats_no_chip=chip_beats,
    )


def chip_report_to_dict(report: ChipReport) -> dict[str, Any]:
    return {
        "season": report.season,
        "event": report.event,
        "best_this_gw": report.best_this_gw,
        "best_no_chip": report.best_no_chip,
        "best_chip": report.best_chip,
        "chip_beats_no_chip": report.chip_beats_no_chip,
        "note": (
            "This is one-week EV, not the option value of saving a chip. "
            "v1 will not auto-play TC/BB/FH/WC."
        ),
        "options": [
            {
                "key": row.key,
                "label": row.label,
                "ev": round(row.ev, 4),
                "gain_vs_hold": round(row.gain_vs_hold, 4),
                "is_chip": row.is_chip,
                "note": row.note,
            }
            for row in report.options
        ],
    }


def format_chip_report(report: ChipReport) -> str:
    season = report.season or "?"
    event = report.event if report.event is not None else "?"
    lines = [
        "Chip EV (this GW only)",
        f"Season {season}  event {event}",
        "",
        f"{'Option':<42} {'EV':>7}  {'vs hold':>8}",
    ]
    for row in report.options:
        mark = " *" if row.key == report.best_this_gw else ""
        lines.append(
            f"{row.label:<42} {row.ev:7.2f}  {row.gain_vs_hold:+8.2f}{mark}"
        )
    lines.extend(
        [
            "",
            f"Best this GW: {report.option(report.best_this_gw).label}",
            f"Best no-chip: {report.option(report.best_no_chip).label}",
            (
                f"Chip beats no-chip this week: {report.chip_beats_no_chip}"
                + (f" ({report.option(report.best_chip).label})" if report.best_chip else "")
            ),
            "",
            "Do not auto-play a chip from this table. Saving TC for a nailed premium DGW "
            "is usually right even if this week’s extra copy looks fine.",
            "FH and WC share this-GW EV; they differ only in whether you keep the squad.",
        ]
    )
    return "\n".join(lines)


def run_chips(
    *,
    settings: Settings | None = None,
    season: str | None = None,
    event: int | None = None,
    squad_path: Path | str | None = None,
    team_id: int | None = None,
    bank_m: float | None = None,
    free_transfers: int = FREE_TRANSFERS_DEFAULT,
    max_transfers: int = MAX_TRANSFERS_DEFAULT,
    captain_min_p_play: float = CAPTAIN_MIN_P_PLAY,
    overlay_live: bool = False,
) -> dict[str, Any]:
    transfer_result = run_transfer(
        settings=settings,
        season=season,
        event=event,
        squad_path=Path(squad_path) if squad_path else None,
        team_id=team_id,
        bank_m=bank_m,
        free_transfers=free_transfers,
        max_transfers=max_transfers,
        captain_min_p_play=captain_min_p_play,
        wildcard=False,
        overlay_live=overlay_live,
    )
    report = evaluate_chips(transfer_result["plan"])
    cfg = settings or load_settings()
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cfg.eval_dir / "chips.json"
    eval_path.write_text(json.dumps(chip_report_to_dict(report), indent=2), encoding="utf-8")
    return {
        "report": report,
        "plan": transfer_result["plan"],
        "eval_path": eval_path,
        "note": transfer_result.get("note", ""),
    }
