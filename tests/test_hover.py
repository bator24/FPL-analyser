from fpl.ui.hover import name_cell, tooltip_text


def test_tooltip_includes_club_value_form_and_next_five() -> None:
    text = tooltip_text(
        {
            "name": "Wilson",
            "team": "WHU",
            "cost_m": 6.5,
            "form": 1.2,
            "points": 4,
            "this_gw": "GW2 CHE (A) FDR5",
            "next_5_text": "GW2 CHE (A) FDR5, GW3 BUR (H) FDR2",
            "fixture_verdict": "Hard fixture ahead (next FDR 5).",
        }
    )
    assert "WHU" in text
    assert "£6.5m" in text
    assert "FPL form 1.2" in text
    assert "CHE (A) FDR5" in text
    assert "Hard fixture ahead" in text


def test_name_cell_escapes_html() -> None:
    html = name_cell({"name": '<img src=x onerror=alert(1)>', "team": "ARS", "cost_m": 5.0})
    assert "<img" not in html
    assert "&lt;img" in html
    assert 'class="fpl-name"' in html
    assert "title=" in html


def test_tooltip_includes_this_week_xg() -> None:
    text = tooltip_text({"name": "Gomez", "e_goals": 0.42, "e_assists": 0.18})
    assert "expected goals 0.42" in text
    assert "expected assists 0.18" in text
    text = tooltip_text({"name": "Gomez", "team": "BHA", "news": "Knock - 75% chance of playing"})
    assert "FPL: Knock" in text


def test_tooltip_includes_this_gw_transfers() -> None:
    text = tooltip_text({"name": "Mbeumo", "transfers_in_event": 124000, "transfers_out_event": 11000})
    assert "in 124,000" in text
    assert "out 11,000" in text
