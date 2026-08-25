import pandas as pd

from fpl.models.horizon import (
    attach_horizon,
    fixture_verdict,
    merge_context,
    next_fixtures_for_team,
    summarize_horizon,
)


def _fixtures() -> pd.DataFrame:
    # Team 1: hard away next, then mixed. Team 2: kind run.
    return pd.DataFrame(
        {
            "fixture_id": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "event": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
            "finished": [True, False, False, False, False, False, False, False, False, False],
            "kickoff_time": [f"2026-08-{i:02d}" for i in range(10, 20)],
            "team_h": [1, 2, 1, 3, 1, 2, 4, 2, 5, 2],
            "team_a": [9, 1, 8, 1, 7, 6, 2, 8, 2, 9],
            "team_h_difficulty": [2, 2, 3, 2, 4, 2, 2, 2, 2, 2],
            "team_a_difficulty": [4, 5, 3, 4, 2, 2, 2, 2, 2, 2],
        }
    )


def _teams() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "short_name": ["ARS", "CHE", "BUR", "BOU", "FUL", "WOL", "EVE", "NEW", "MCI"],
        }
    )


def test_next_fixtures_skip_finished_and_use_side_fdr() -> None:
    items = next_fixtures_for_team(_fixtures(), 1, from_event=1, n=5, teams=_teams())
    assert len(items) == 4
    assert items[0]["label"] == "GW2 CHE (A) FDR5"
    assert items[0]["short"] == "CHE(A)5"
    assert items[0]["fdr"] == 5
    assert items[1]["label"] == "GW3 NEW (H) FDR3"


def test_hard_fixture_ahead_verdict() -> None:
    items = next_fixtures_for_team(_fixtures(), 1, from_event=2, n=5, teams=_teams())
    summary = summarize_horizon(items)
    assert "Hard fixture ahead" in summary["fixture_verdict"]
    assert summary["next_fdr"] == 5
    assert "CHE(A)5" in summary["next_5_short"]


def test_kind_run_verdict() -> None:
    items = next_fixtures_for_team(_fixtures(), 2, from_event=2, n=5, teams=_teams())
    summary = summarize_horizon(items)
    assert "Kind run" in summary["fixture_verdict"]
    assert summary["hard_n"] == 0


def test_attach_horizon_maps_team() -> None:
    players = pd.DataFrame(
        {
            "element_id": [411, 1],
            "web_name": ["Haaland", "Raya"],
            "team_id": [9, 1],
            "cost_m": [14.5, 5.5],
            "form": [0.0, 2.0],
        }
    )
    out = attach_horizon(players, _fixtures(), _teams(), from_event=2)
    ars = out.loc[out["element_id"] == 1].iloc[0]
    assert "CHE(A)5" in str(ars["next_5_short"])
    assert "Hard fixture ahead" in str(ars["fixture_verdict"])


def test_fixture_verdict_blank() -> None:
    assert "No upcoming" in fixture_verdict(fdr_mean=None, hard_n=0, next_fdr=None, n=0)


def test_merge_context_fills_horizon() -> None:
    row = {"element_id": 1, "name": "Wilson", "xpts": 1.98, "p_play": 1.0}
    ctx = {
        "team": "WHU",
        "cost_m": 6.5,
        "form": 1.2,
        "next_5_text": "GW2 CHE (A) FDR5, GW3 BUR (H) FDR2",
        "fixture_verdict": "Hard fixture ahead (next FDR 5). Hard run (3 of 5 at FDR 4+; mean 4.0).",
        "this_gw": "GW2 CHE (A) FDR5",
    }
    merged = merge_context(row, ctx)
    assert merged["name"] == "Wilson"
    assert merged["team"] == "WHU"
    assert "Hard fixture ahead" in merged["fixture_verdict"]
    assert merged["form"] == 1.2
