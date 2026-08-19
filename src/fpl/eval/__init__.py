"""Always-on baseline evaluation (never optional)."""

from fpl.eval.baselines import brier, mae, minutes_eval_table
from fpl.eval.xpts import xpts_eval_table

__all__ = ["mae", "brier", "minutes_eval_table", "xpts_eval_table"]
