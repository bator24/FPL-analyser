from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fpl.config import Settings, load_settings
from fpl.eval.baselines import minutes_eval_table
from fpl.eval.xpts import xpts_eval_table
from fpl.models.minutes import (
    DEFAULT_TEST_SEASONS,
    MinutesModel,
    apply_xmins_overrides,
    baseline_minutes,
    walk_forward_predict,
)
from fpl.models.xpts import walk_forward_xpts


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


def run_xpts(
    *,
    settings: Settings | None = None,
    test_seasons: tuple[str, ...] = DEFAULT_TEST_SEASONS,
) -> dict[str, Any]:
    cfg = settings or load_settings()
    panel_path = cfg.processed_dir / "player_gw.parquet"
    if not panel_path.exists():
        raise RuntimeError("Missing player_gw.parquet. Run `python -m fpl history` first.")
    panel = pd.read_parquet(panel_path)
    print(f"Walk-forward xPts on {len(panel)} rows...", flush=True)
    oos = walk_forward_xpts(panel, test_seasons=test_seasons)
    actual = pd.to_numeric(oos["total_points"], errors="coerce")
    baselines = {
        "total_points_r5": pd.to_numeric(oos["total_points_r5"], errors="coerce").fillna(0),
        "total_points_lag1": pd.to_numeric(oos.get("total_points_lag1"), errors="coerce").fillna(0),
        "structural": pd.to_numeric(oos["xpts_structural"], errors="coerce"),
        "blend": 0.65 * pd.to_numeric(oos["xpts_structural"], errors="coerce")
        + 0.35 * pd.to_numeric(oos["total_points_r5"], errors="coerce").fillna(0),
        "fpl_xp_posthoc": pd.to_numeric(oos.get("fpl_xp_posthoc"), errors="coerce").fillna(0),
    }
    report = xpts_eval_table(
        actual_points=actual,
        model_points=pd.to_numeric(oos["xpts"], errors="coerce"),
        baselines=baselines,
        positions=oos.get("position"),
    )
    report["test_seasons"] = sorted(oos["season"].astype(str).unique().tolist())
    report["note"] = (
        "xpts is the structural FPL scoring model (minutes × rates + Poisson CS + BPS/bonus). "
        "fpl_xp_posthoc is vaastav's post-match scrape — not a pre-deadline feature."
    )
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        c
        for c in [
            "season",
            "event",
            "element_id",
            "name",
            "position",
            "team",
            "total_points",
            "minutes",
            "xpts",
            "xpts_structural",
            "xpts_r5",
            "e_minutes",
            "p_play",
            "p_60",
            "e_goals",
            "e_assists",
            "p_cs",
            "e_bonus",
            "e_bps",
            "fold",
        ]
        if c in oos.columns
    ]
    oos[keep].to_parquet(cfg.processed_dir / "xpts_oos.parquet", index=False)
    (cfg.eval_dir / "xpts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"oos": oos, "report": report, "eval_path": cfg.eval_dir / "xpts.json"}


def format_xpts_report(result: dict[str, Any]) -> str:
    report = result["report"]
    model = report["model"]
    r5 = report["baselines"]["total_points_r5"]
    lag1 = report["baselines"]["total_points_lag1"]
    structural = report["baselines"]["structural"]
    blend = report["baselines"].get("blend", structural)
    xp = report["baselines"]["fpl_xp_posthoc"]
    lines = [
        "Expected points (season walk-forward)",
        f"Test seasons: {', '.join(report['test_seasons'])}",
        f"Rows: {report['n']}",
        "",
        "MAE points (lower is better):",
        f"  shipped     {model['mae']:.3f}  Spearman {model['spearman']:.3f}",
        f"  last-5      {r5['mae']:.3f}  Spearman {r5['spearman']:.3f}   MAE gap {model['mae'] - r5['mae']:+.3f}",
        f"  last-1      {lag1['mae']:.3f}",
        f"  blend       {blend['mae']:.3f}  Spearman {blend['spearman']:.3f}",
        f"  FPL xP*     {xp['mae']:.3f}  (*post-match scrape, leaky)",
        "",
        "RMSE by actual return bucket:",
    ]
    for bucket in ("zeros", "blanks", "tickers", "haulers"):
        m = model["buckets"][bucket]
        b = r5["buckets"][bucket]
        lines.append(
            f"  {bucket:8} n={m['n']:<7} model {m['rmse']:.3f}  r5 {b['rmse']:.3f}"
        )
    if report.get("by_position"):
        lines.append("\nBy position (shipped):")
        for pos, row in report["by_position"].items():
            lines.append(f"  {pos}  MAE {row['mae']:.3f}  Spearman {row['spearman']:.3f}  n={row['n']}")
    lines.extend(
        [
            "",
            f"Beats last-5 MAE: {report['beats_r5_mae']}",
            f"Beats last-5 Spearman: {report['beats_r5_spearman']}",
            f"Kill criterion (both): {report['kill_pass']}",
            report.get("note", ""),
            f"Eval file: {result['eval_path']}",
        ]
    )
    return "\n".join(lines)
