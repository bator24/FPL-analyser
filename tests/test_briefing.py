import pandas as pd

from fpl.advisor import briefing as briefing_mod
from fpl.advisor.briefing import format_briefing, last_finished_event, recap_from_panel
from fpl.advisor.chat import local_reply


def test_briefing_imports_chip_formatters() -> None:
    assert callable(briefing_mod.format_chip_report)
    assert callable(briefing_mod.run_chips)


def test_last_finished_event() -> None:
    fixtures = pd.DataFrame({"event": [1, 1, 2], "finished": [True, True, False]})
    assert last_finished_event(fixtures) == 1


def test_recap_sums_and_flags_blanks() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "event": [38, 38, 38],
            "element_id": [1, 1, 2],
            "name": ["Salah", "Salah", "Nunez"],
            "position": ["MID", "MID", "FWD"],
            "team": ["LIV", "LIV", "LIV"],
            "minutes": [90, 30, 0],
            "total_points": [8, 2, 0],
        }
    )
    recap = recap_from_panel(panel, season="2025-26", event=38, squad_ids=[1, 2, 99])
    assert recap["total_points"] == 10
    salah = next(p for p in recap["players"] if p["name"] == "Salah")
    assert salah["minutes"] == 120
    assert salah["points"] == 10
    assert "Nunez" in recap["did_not_play"]
    assert 99 in recap["missing_ids"]


def test_format_briefing_hold_and_captain() -> None:
    recap = {
        "headline": "No 2026-27 matches yet. Last completed PL: 2025-26 GW38",
        "note": "Raw combined points.",
        "players": [
            {
                "name": "Salah",
                "position": "MID",
                "minutes": 90,
                "points": 12,
            }
        ],
        "n_found": 1,
        "total_points": 12,
        "did_not_play": [],
        "blanks": [],
        "best": {"name": "Salah", "points": 12, "minutes": 90},
        "missing_ids": [],
    }
    upcoming = {
        "season": "2026-27",
        "event": 1,
        "action": "HOLD",
        "expected_net": 0.1,
        "hold_ev": 50.0,
        "n_transfers": 0,
        "hits": 0,
        "transfers_out": [],
        "transfers_in": [],
        "captain_hold": {"name": "Mbeumo", "xpts": 7.1, "p_play": 1.0},
        "chip_beats_no_chip": False,
        "best_no_chip": "hold",
        "chip_best_label": "hold",
        "chip_note": "Do not auto-play.",
        "engine_note": "Live prior.",
    }
    text = format_briefing(recap, upcoming)
    assert "**HOLD.**" in text
    assert "Mbeumo" in text
    assert "do not play one" in text.lower() or "Do not play" in text or "do not play" in text


def test_local_reply_routes_captain_and_hold() -> None:
    facts = {
        "recap": {
            "headline": "2025-26 GW38",
            "best": {"name": "Salah", "points": 12, "minutes": 90},
            "did_not_play": ["Nunez"],
            "players": [{"name": "Salah", "points": 12, "minutes": 90, "position": "MID"}],
        },
        "upcoming": {
            "action": "HOLD",
            "expected_net": -0.4,
            "hold_ev": 48.2,
            "captain_hold": {"name": "Mbeumo", "xpts": 7.14, "p_play": 1.0},
            "chip_beats_no_chip": False,
            "best_no_chip": "hold",
            "chip_note": "this-GW only",
            "xi": [{"name": "Mbeumo", "xpts": 7.14, "p_play": 1.0, "captain": True}],
        },
        "flags": [],
    }
    cap = local_reply(facts, "who should I captain?")
    assert "Mbeumo" in cap
    hold = local_reply(facts, "should I transfer this week?")
    assert "HOLD" in hold
    last = local_reply(facts, "how did last week go?")
    assert "Salah" in last or "GW38" in last
    named = local_reply(facts, "tell me about Mbeumo")
    assert "7.14" in named
