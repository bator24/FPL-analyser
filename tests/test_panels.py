from fpl.ui.panels import _player_side_cells


def test_incoming_side_shows_club_value_form_and_next_five() -> None:
    cells = _player_side_cells(
        {
            "name": "Gomez",
            "team": "BHA",
            "cost_m": 5.0,
            "xpts": 7.3,
            "p_play": 1.0,
            "form": 6.0,
            "e_goals": 0.42,
            "e_assists": 0.18,
            "next_5_short": "CHE(A)4 · LEE(H)2",
            "news": "Knock - 75% chance of playing",
        },
        None,
    )
    blob = " ".join(cells)
    assert "Gomez" in blob
    assert "BHA" in blob
    assert "£5.0m" in blob
    assert "7.30" in blob
    assert "6.0" in blob
    assert "CHE(A)4" in blob
    assert "Knock" in blob
    assert "0.42" in blob
    assert "0.18" in blob
    assert len(cells) == 10
