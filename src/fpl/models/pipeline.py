from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fpl.config import Settings, load_settings
from fpl.eval.baselines import minutes_eval_table
from fpl.models.minutes import (
    DEFAULT_TEST_SEASONS,
    MinutesModel,
    apply_xmins_overrides,
    baseline_minutes,
    walk_forward_predict,
)


def _load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_minutes(
    *,
    settings: Settings | None = None,
    test_seasons: tuple[str, ...] = DEFAULT_TEST_SEASONS,
) -> dict[str, Any]:
    cfg = settings or load_settings()
    panel_path = cfg.processed_dir / "player_gw.parquet"
    if not panel_path.exists():
        raise RuntimeError("Missing player_gw.parquet. Run `python -m fpl history` first.")
    panel = pd.read_parquet(panel_path)
    print(f"Walk-forward minutes model on {len(panel)} rows...", flush=True)
    oos = walk_forward_predict(panel, test_seasons=test_seasons)
    overrides = _load_overrides(cfg.overrides_dir / "xmins.csv")
    oos = apply_xmins_overrides(oos, overrides, current_season=cfg.current_season)
    baselines = baseline_minutes(oos)
    report = minutes_eval_table(
        actual_minutes=pd.to_numeric(oos["minutes"], errors="coerce"),
        model_minutes=pd.to_numeric(oos["e_minutes"], errors="coerce"),
        p_play=pd.to_numeric(oos["p_play"], errors="coerce"),
        baselines=baselines,
    )
    report["test_seasons"] = sorted(oos["season"].astype(str).unique().tolist())
    report["n_overrides"] = int(oos["override"].fillna(False).sum()) if "override" in oos.columns else 0

    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        c
        for c in [
            "season",
            "event",
            "element_id",
            "name",
            "position",
            "team",
            "minutes",
            "played",
            "p_play",
            "p_zero",
            "e_minutes",
            "p_60",
            "minutes_if_play",
            "confidence",
            "override",
            "fold",
            "minutes_r3",
            "minutes_lag1",
        ]
        if c in oos.columns
    ]
    oos[keep].to_parquet(cfg.processed_dir / "minutes_oos.parquet", index=False)
    (cfg.eval_dir / "minutes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Fitting full-history model for later inference...", flush=True)
    from fpl.models.minutes import prepare_minutes_frame

    full = MinutesModel.unfitted().fit(prepare_minutes_frame(panel))
    joblib.dump(full, cfg.models_dir / "minutes.joblib")
    return {"oos": oos, "report": report, "eval_path": cfg.eval_dir / "minutes.json"}


def format_minutes_report(result: dict[str, Any]) -> str:
    report = result["report"]
    model = report["model"]
    r3 = report["baselines"]["minutes_r3"]
    lag1 = report["baselines"]["minutes_lag1"]
    heuristic = report["baselines"]["starter_heuristic"]
    sticky = report["baselines"].get("sticky", heuristic)
    lines = [
        "Minutes model (season walk-forward, out-of-sample)",
        f"Test seasons: {', '.join(report['test_seasons'])}",
        f"Rows: {report['n']}  zeros: {report['n_zero']} ({report['zero_rate']:.1%})",
        "",
        "MAE minutes (lower is better):",
        f"  model     {model['mae']:.2f}",
        f"  r3        {r3['mae']:.2f}   gap {model['mae'] - r3['mae']:+.2f}",
        f"  lag1      {lag1['mae']:.2f}",
        f"  sticky    {sticky['mae']:.2f}",
        f"  heuristic {heuristic['mae']:.2f}",
        "",
        "MAE on actual zeros (OpenFPL's weak spot):",
        f"  model     {model['mae_zeros']:.2f}",
        f"  r3        {r3['mae_zeros']:.2f}   gap {model['mae_zeros'] - r3['mae_zeros']:+.2f}",
        f"  lag1      {lag1['mae_zeros']:.2f}",
        "",
        f"Brier (played): {model['brier_played']:.3f}",
        f"Mean p_play | zero:   {model['mean_p_play_on_zeros']:.3f}",
        f"Mean p_play | played: {model['mean_p_play_on_played']:.3f}",
        "",
        f"Beats r3 overall: {report['beats_r3_mae']}",
        f"Beats r3 on zeros: {report['beats_r3_mae_zeros']}",
        f"Kill criterion (both): {report['kill_pass']}",
        "Note: e_minutes is last-GW sticky prior; a minutes GBM lost to it on this panel.",
        f"Eval file: {result['eval_path']}",
    ]
    return "\n".join(lines)
