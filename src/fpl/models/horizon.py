"""Next-N unfinished PL fixtures and official FPL FDR (1 easiest, 5 hardest).

Not a news scrape and not a multi-week xPts model. This is the FPL fixture list
plus `team_h_difficulty` / `team_a_difficulty` so the advisor can say "hard
fixture ahead" when that is actually in the snapshot.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.config import Settings, load_settings

HORIZON_N = 5
HARD_FDR = 4

CONTEXT_KEYS = (
    "name",
    "team",
    "team_id",
    "cost_m",
    "form",
    "points_per_game",
    "total_points",
    "this_gw",
    "next_5_short",
    "next_5_text",
    "fdr_mean",
    "hard_n",
    "next_fdr",
    "fixture_verdict",
)


def _num(value: Any) -> float | None:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(n):
        return None
    return float(n)


def team_short_map(teams: pd.DataFrame | None) -> dict[int, str]:
    if teams is None or teams.empty or "team_id" not in teams.columns:
        return {}
    name_col = "short_name" if "short_name" in teams.columns else "name"
    if name_col not in teams.columns:
        return {}
    out: dict[int, str] = {}
    for tid, name in zip(
        pd.to_numeric(teams["team_id"], errors="coerce"),
        teams[name_col],
    ):
        if pd.isna(tid):
            continue
        out[int(tid)] = str(name)
    return out


def next_fixtures_for_team(
    fixtures: pd.DataFrame,
    team_id: int,
    *,
    from_event: int,
    n: int = HORIZON_N,
    teams: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Unfinished matches at event >= from_event, oldest first, up to n fixtures."""
    if fixtures is None or fixtures.empty:
        return []
    tid = int(team_id)
    names = team_short_map(teams)
    fx = fixtures.copy()
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    for col in ("team_h", "team_a", "team_h_difficulty", "team_a_difficulty"):
        if col in fx.columns:
            fx[col] = pd.to_numeric(fx[col], errors="coerce")
    fx = fx.loc[fx["event"].notna() & (fx["event"] >= int(from_event))]
    if "finished" in fx.columns:
        fx = fx.loc[~fx["finished"].fillna(False).astype(bool)]
    home = fx["team_h"] == tid if "team_h" in fx.columns else False
    away = fx["team_a"] == tid if "team_a" in fx.columns else False
    fx = fx.loc[home | away]
    sort_cols = [c for c in ("event", "kickoff_time", "fixture_id") if c in fx.columns]
    if sort_cols:
        fx = fx.sort_values(sort_cols, kind="mergesort")
    items: list[dict[str, Any]] = []
    for _, row in fx.head(int(n)).iterrows():
        is_home = int(row.get("team_h") or 0) == tid
        opp_id = row.get("team_a") if is_home else row.get("team_h")
        opp_n = _num(opp_id)
        opp_id_i = int(opp_n) if opp_n is not None else None
        opp = names.get(opp_id_i, str(opp_id_i or "?"))
        fdr_n = _num(row.get("team_h_difficulty") if is_home else row.get("team_a_difficulty"))
        fdr = int(fdr_n) if fdr_n is not None else None
        side = "H" if is_home else "A"
        event = int(row["event"])
        fdr_s = str(fdr) if fdr is not None else "?"
        items.append(
            {
                "event": event,
                "home": is_home,
                "opponent_id": opp_id_i,
                "opponent": opp,
                "fdr": fdr,
                "label": f"GW{event} {opp} ({side}) FDR{fdr_s}",
                "short": f"{opp}({side}){fdr_s}",
            }
        )
    return items


def fixture_verdict(
    *,
    fdr_mean: float | None,
    hard_n: int,
    next_fdr: int | None,
    n: int,
) -> str:
    if n <= 0:
        return "No upcoming PL fixtures in the snapshot."
    next_hard = next_fdr is not None and next_fdr >= HARD_FDR
    mean_s = f"{fdr_mean:.1f}" if fdr_mean is not None else "?"
    if hard_n >= 3 or (fdr_mean is not None and fdr_mean >= 3.7) or (next_hard and hard_n >= 2):
        msg = f"Hard run ({hard_n} of {n} at FDR {HARD_FDR}+; mean {mean_s})."
        if next_hard:
            return f"Hard fixture ahead (next FDR {int(next_fdr)}). {msg}"
        return msg
    if next_hard:
        return (
            f"Hard fixture ahead (next FDR {int(next_fdr)}); "
            f"the rest of the {n} is mixed (mean {mean_s})."
        )
    if fdr_mean is not None and fdr_mean <= 2.5 and hard_n <= 1:
        return f"Kind run (mean FDR {mean_s}; {hard_n} of {n} at FDR {HARD_FDR}+)."
    return f"Mixed run ({hard_n} of {n} at FDR {HARD_FDR}+; mean {mean_s})."


def summarize_horizon(items: list[dict[str, Any]]) -> dict[str, Any]:
    fdrs = [int(x["fdr"]) for x in items if x.get("fdr") is not None]
    hard_n = sum(1 for f in fdrs if f >= HARD_FDR)
    fdr_mean = round(sum(fdrs) / len(fdrs), 2) if fdrs else None
    next_fdr = items[0]["fdr"] if items else None
    this_gw = str(items[0]["label"]) if items else ""
    return {
        "this_gw": this_gw,
        "next_5_short": " · ".join(str(x["short"]) for x in items),
        "next_5_text": ", ".join(str(x["label"]) for x in items),
        "fdr_mean": fdr_mean,
        "hard_n": hard_n,
        "next_fdr": next_fdr,
        "fixture_verdict": fixture_verdict(
            fdr_mean=fdr_mean, hard_n=hard_n, next_fdr=next_fdr, n=len(items)
        ),
    }


def horizon_by_team(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame | None,
    *,
    from_event: int,
    n: int = HORIZON_N,
    team_ids: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    ids = team_ids
    if ids is None:
        names = team_short_map(teams)
        ids = sorted(names)
        if not ids and fixtures is not None and not fixtures.empty:
            found: set[int] = set()
            for col in ("team_h", "team_a"):
                if col in fixtures.columns:
                    found.update(
                        int(v)
                        for v in pd.to_numeric(fixtures[col], errors="coerce").dropna()
                    )
            ids = sorted(found)
    out: dict[int, dict[str, Any]] = {}
    for tid in ids:
        out[int(tid)] = summarize_horizon(
            next_fixtures_for_team(
                fixtures, int(tid), from_event=from_event, n=n, teams=teams
            )
        )
    return out


def attach_horizon(
    players: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    teams: pd.DataFrame | None,
    *,
    from_event: int,
    n: int = HORIZON_N,
) -> pd.DataFrame:
    """Copy next-N FDR columns onto player rows via `team_id`."""
    out = players.copy()
    cols = (
        "this_gw",
        "next_5_short",
        "next_5_text",
        "fdr_mean",
        "hard_n",
        "next_fdr",
        "fixture_verdict",
    )
    if out.empty or fixtures is None or fixtures.empty or "team_id" not in out.columns:
        for col in cols:
            if col not in out.columns:
                out[col] = pd.NA if col in {"fdr_mean", "hard_n", "next_fdr"} else ""
        return out
    tids = [
        int(v)
        for v in pd.to_numeric(out["team_id"], errors="coerce").dropna().unique()
    ]
    by_team = horizon_by_team(
        fixtures, teams, from_event=from_event, n=n, team_ids=tids
    )
    extra = pd.DataFrame(
        [{"team_id": tid, **payload} for tid, payload in by_team.items()]
    )
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce")
    if extra.empty:
        for col in cols:
            if col not in out.columns:
                out[col] = pd.NA if col in {"fdr_mean", "hard_n", "next_fdr"} else ""
        return out
    extra["team_id"] = pd.to_numeric(extra["team_id"], errors="coerce")
    out = out.drop(columns=[c for c in cols if c in out.columns])
    return out.merge(extra, on="team_id", how="left")


def _player_name(row: pd.Series) -> str:
    for col in ("web_name", "name"):
        val = row.get(col)
        if val is not None and str(val).strip() and str(val) != "nan":
            return str(val)
    return str(row.get("element_id") or "?")


def context_from_attached(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if frame is None or frame.empty or "element_id" not in frame.columns:
        return {}
    work = frame.copy()
    work["element_id"] = pd.to_numeric(work["element_id"], errors="coerce")
    work = work.dropna(subset=["element_id"]).drop_duplicates("element_id")
    work["element_id"] = work["element_id"].astype(int)
    out: dict[int, dict[str, Any]] = {}
    for _, row in work.iterrows():
        eid = int(row["element_id"])
        team = row.get("team")
        if team is None or (isinstance(team, float) and pd.isna(team)) or str(team) in {"", "nan", "<NA>"}:
            team = ""
        ctx: dict[str, Any] = {
            "element_id": eid,
            "name": _player_name(row),
            "team": "" if team is None or str(team) == "nan" else str(team),
            "team_id": None if _num(row.get("team_id")) is None else int(_num(row.get("team_id"))),
            "cost_m": _num(row.get("cost_m")),
            "form": _num(row.get("form")),
            "points_per_game": _num(row.get("points_per_game")),
            "total_points": _num(row.get("total_points") if pd.notna(row.get("total_points")) else row.get("points")),
            "this_gw": str(row.get("this_gw") or ""),
            "next_5_short": str(row.get("next_5_short") or ""),
            "next_5_text": str(row.get("next_5_text") or ""),
            "fdr_mean": _num(row.get("fdr_mean")),
            "hard_n": None if _num(row.get("hard_n")) is None else int(_num(row.get("hard_n"))),
            "next_fdr": None if _num(row.get("next_fdr")) is None else int(_num(row.get("next_fdr"))),
            "fixture_verdict": str(row.get("fixture_verdict") or ""),
        }
        out[eid] = ctx
    return out


def load_player_context(
    settings: Settings | None = None,
    *,
    from_event: int,
    n: int = HORIZON_N,
) -> dict[int, dict[str, Any]]:
    cfg = settings or load_settings()
    players_path = cfg.processed_dir / "players.parquet"
    fixtures_path = cfg.processed_dir / "fixtures.parquet"
    teams_path = cfg.processed_dir / "teams.parquet"
    if not players_path.exists() or not fixtures_path.exists():
        return {}
    players = pd.read_parquet(players_path)
    fixtures = pd.read_parquet(fixtures_path)
    teams = pd.read_parquet(teams_path) if teams_path.exists() else pd.DataFrame()
    if not teams.empty and "team_id" in teams.columns and "short_name" in teams.columns:
        if "team" not in players.columns:
            players = players.merge(teams[["team_id", "short_name"]], on="team_id", how="left")
            players["team"] = players["short_name"]
    attached = attach_horizon(players, fixtures, teams, from_event=from_event, n=n)
    return context_from_attached(attached)


def merge_context(row: dict[str, Any], ctx: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    if not ctx:
        return out
    for key in CONTEXT_KEYS:
        incoming = ctx.get(key)
        if incoming in (None, "", []):
            continue
        try:
            if pd.isna(incoming):
                continue
        except (TypeError, ValueError):
            pass
        current = out.get(key)
        overlay = key not in {"name", "cost_m", "position", "xpts", "p_play", "element_id"}
        if overlay or current in (None, "", []):
            if key == "name" and current not in (None, "", []):
                continue
            if key == "cost_m" and current not in (None, ""):
                continue
            out[key] = incoming
    return out


def enrich_rows(rows: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        eid = row.get("element_id")
        try:
            key = int(eid) if eid is not None else None
        except (TypeError, ValueError):
            key = None
        out.append(merge_context(row, by_id.get(key) if key is not None else None))
    return out
