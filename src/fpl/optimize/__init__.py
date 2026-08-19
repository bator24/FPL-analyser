"""PuLP squad, transfer, and chip solvers (Phases 4–6)."""

from fpl.optimize.rules import is_legal, legality_errors
from fpl.optimize.squad import brute_force_squad, solve_squad
from fpl.optimize.transfers import solve_transfers
from fpl.optimize.chips import evaluate_chips

__all__ = [
    "solve_squad",
    "brute_force_squad",
    "solve_transfers",
    "evaluate_chips",
    "is_legal",
    "legality_errors",
]
