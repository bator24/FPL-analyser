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


def test_format_briefing_take_is_spoken_english() -> None:
    recap = {
        "headline": "GW1",
        "note": "",
        "players": [],
        "n_found": 0,
        "total_points": 0,
        "did_not_play": [],
        "blanks": [],
        "best": None,
        "missing_ids": [],
    }
    upcoming = {
        "season": "2026-27",
        "event": 2,
        "action": "TAKE TRANSFERS",
        "expected_net": 7.88,
        "hold_ev": 53.41,
        "chosen_ev": 61.29,
        "n_transfers": 1,
        "hits": 1,
        "transfers_out": [
            {
                "name": "Wilson",
                "position": "MID",
                "cost_m": 6.5,
                "xpts": 1.98,
                "p_play": 1.0,
                "team": "WHU",
                "form": 1.2,
                "this_gw": "GW2 CHE (A) FDR5",
                "next_5_text": "GW2 CHE (A) FDR5, GW3 BUR (H) FDR2",
                "fixture_verdict": "Hard fixture ahead (next FDR 5). Hard run (1 of 2 at FDR 4+; mean 3.5).",
                "next_fdr": 5,
            }
        ],
        "transfers_in": [
            {
                "name": "Gomez",
                "position": "MID",
                "cost_m": 5.0,
                "xpts": 4.50,
                "p_play": 1.0,
                "team": "BRI",
                "form": 4.0,
                "this_gw": "GW2 WOL (H) FDR2",
                "next_5_text": "GW2 WOL (H) FDR2",
                "fixture_verdict": "Kind run (mean FDR 2.0; 0 of 1 at FDR 4+).",
                "next_fdr": 2,
            }
        ],
        "captain_hold": {"name": "Haaland", "xpts": 7.0, "p_play": 1.0},
        "chip_beats_no_chip": False,
        "best_no_chip": "transfers",
        "chip_note": "Do not auto-play.",
        "engine_note": "",
    }
    text = format_briefing(recap, upcoming)
    assert "I'd sell Wilson" in text or "I'd sell Wilson (WHU" in text
    assert "Take them" in text
    assert "xPts" not in text
    assert "p_play" not in text
    assert "| Out |" not in text
    assert "away at CHE" in text
    assert "Haaland" in text


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


def test_local_reply_explains_why_player_is_out() -> None:
    facts = {
        "recap": {
            "players": [{"name": "Wilson", "points": 2, "minutes": 65, "position": "MID"}],
        },
        "upcoming": {
            "action": "TAKE TRANSFERS",
            "expected_net": 7.88,
            "hold_ev": 53.41,
            "chosen_ev": 61.29,
            "n_transfers": 2,
            "hits": 1,
            "transfers_out": [
                {
                    "name": "Anderson",
                    "position": "MID",
                    "cost_m": 5.5,
                    "xpts": 2.10,
                    "p_play": 0.75,
                    "team": "NFO",
                    "form": 2.0,
                    "this_gw": "GW2 BUR (H) FDR2",
                    "next_5_text": "GW2 BUR (H) FDR2, GW3 BOU (A) FDR2",
                    "next_5_short": "BUR(H)2 · BOU(A)2",
                    "fixture_verdict": "Kind run (mean FDR 2.0; 0 of 2 at FDR 4+).",
                    "next_fdr": 2,
                },
                {
                    "name": "Wilson",
                    "position": "MID",
                    "cost_m": 6.5,
                    "xpts": 1.98,
                    "p_play": 1.0,
                    "team": "WHU",
                    "form": 1.2,
                    "this_gw": "GW2 CHE (A) FDR5",
                    "next_5_text": "GW2 CHE (A) FDR5, GW3 BUR (H) FDR2, GW4 ARS (A) FDR4, GW5 BOU (H) FDR2, GW6 MCI (A) FDR5",
                    "next_5_short": "CHE(A)5 · BUR(H)2 · ARS(A)4 · BOU(H)2 · MCI(A)5",
                    "fixture_verdict": "Hard fixture ahead (next FDR 5). Hard run (3 of 5 at FDR 4+; mean 3.6).",
                    "next_fdr": 5,
                    "hard_n": 3,
                    "fdr_mean": 3.6,
                },
            ],
            "transfers_in": [
                {
                    "name": "Mbeumo",
                    "position": "MID",
                    "cost_m": 8.0,
                    "xpts": 8.20,
                    "p_play": 1.0,
                    "team": "MUN",
                    "form": 8.0,
                    "this_gw": "GW2 FUL (H) FDR2",
                    "next_5_text": "GW2 FUL (H) FDR2, GW3 BUR (A) FDR2",
                    "next_5_short": "FUL(H)2 · BUR(A)2",
                    "fixture_verdict": "Kind run (mean FDR 2.0; 0 of 2 at FDR 4+).",
                    "next_fdr": 2,
                },
                {
                    "name": "Gomez",
                    "position": "MID",
                    "cost_m": 5.0,
                    "xpts": 4.50,
                    "p_play": 1.0,
                    "team": "BRI",
                    "form": 4.0,
                    "this_gw": "GW2 WOL (H) FDR2",
                    "next_5_text": "GW2 WOL (H) FDR2, GW3 EVE (A) FDR2",
                    "next_5_short": "WOL(H)2 · EVE(A)2",
                    "fixture_verdict": "Kind run (mean FDR 2.0; 0 of 2 at FDR 4+).",
                    "next_fdr": 2,
                },
            ],
            "captain_hold": {"name": "Haaland", "xpts": 7.0, "p_play": 1.0},
            "xi": [],
            "chip_beats_no_chip": False,
            "best_no_chip": "transfers",
            "chip_note": "this-GW only",
            "engine_note": "",
        },
        "flags": [
            {
                "name": "Anderson",
                "status": "d",
                "news": "Knock - 75% chance of playing",
                "chance_of_playing_next_round": 75.0,
            }
        ],
    }
    text = local_reply(facts, "why do you want to transfer out wilson")
    assert "Wilson" in text
    assert "1.98" in text
    assert "Take them" in text
    assert "You asked about Wilson" in text
    assert "away at CHE" in text
    assert "grim stretch" in text
    assert "recent form" in text
    assert "Last time out Wilson" in text
    assert "argument for selling Wilson" in text
    assert "xPts" not in text
    assert "p_play" not in text
    assert "Knock" in local_reply(facts, "why anderson")


def test_local_reply_is_honest_when_fixtures_are_kind() -> None:
    facts = {
        "recap": {"players": []},
        "upcoming": {
            "action": "TAKE TRANSFERS",
            "expected_net": 1.2,
            "hold_ev": 50.0,
            "chosen_ev": 51.2,
            "n_transfers": 1,
            "hits": 0,
            "transfers_out": [
                {
                    "name": "Palmer",
                    "position": "MID",
                    "cost_m": 10.5,
                    "xpts": 4.0,
                    "p_play": 1.0,
                    "team": "CHE",
                    "form": 5.0,
                    "this_gw": "GW2 BUR (H) FDR2",
                    "next_5_text": "GW2 BUR (H) FDR2, GW3 WOL (A) FDR2",
                    "fixture_verdict": "Kind run (mean FDR 2.0; 0 of 2 at FDR 4+).",
                    "next_fdr": 2,
                }
            ],
            "transfers_in": [
                {
                    "name": "Saka",
                    "position": "MID",
                    "cost_m": 10.0,
                    "xpts": 5.5,
                    "p_play": 1.0,
                    "team": "ARS",
                    "form": 6.0,
                    "this_gw": "GW2 MCI (A) FDR5",
                    "next_5_text": "GW2 MCI (A) FDR5, GW3 LIV (H) FDR4",
                    "fixture_verdict": "Hard fixture ahead (next FDR 5). Hard run (2 of 2 at FDR 4+; mean 4.5).",
                    "next_fdr": 5,
                }
            ],
            "captain_hold": {"name": "Haaland", "xpts": 7.0, "p_play": 1.0},
            "xi": [],
            "chip_beats_no_chip": False,
            "best_no_chip": "transfers",
            "chip_note": "this-GW only",
            "engine_note": "",
        },
        "flags": [],
    }
    text = local_reply(facts, "why sell palmer")
    assert "I would not sell Palmer because of fixtures" in text
    assert "not a fixture punt" in text
