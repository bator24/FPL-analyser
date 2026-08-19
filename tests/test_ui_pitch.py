import pandas as pd

from fpl.ui.pitch import (
    empty_slots,
    fill_from_ids,
    filter_candidates,
    flatten_slots,
    n_filled,
    spent_m,
    would_break_club_cap,
)


def _catalog() -> pd.DataFrame:
    rows = []
    eid = 1
    for pos, n, team in [("GKP", 3, "ARS"), ("DEF", 6, "LIV"), ("MID", 6, "MCI"), ("FWD", 4, "CHE")]:
        for i in range(n):
            rows.append(
                {
                    "element_id": eid,
                    "name": f"{pos}{i}",
                    "position": pos,
                    "team": team if i < 3 else "NEW",
                    "team_id": 1 if i < 3 else 2,
                    "cost_m": 4.5 + i,
                    "points": 10 * i,
                    "form": float(i),
                    "selected_by_percent": 5.0 * i,
                }
            )
            eid += 1
    return pd.DataFrame(rows)


def test_fill_from_ids_respects_position_caps() -> None:
    cat = _catalog()
    gkp = cat.loc[cat["position"] == "GKP", "element_id"].tolist()
    extra = gkp[:3]
    slots = fill_from_ids(extra, cat)
    assert n_filled(slots) == 2
    assert slots["GKP"][0] == extra[0]
    assert None not in slots["GKP"]


def test_filter_is_position_and_name_and_price() -> None:
    cat = _catalog()
    mids = filter_candidates(cat, position="MID", exclude_ids=set(), name="MID1", min_cost=5.0, max_cost=6.0)
    assert set(mids["name"]) == {"MID1"}
    cheap = filter_candidates(cat, position="FWD", exclude_ids=set(), max_cost=4.5, sort_by="Value (cheap)")
    assert cheap.iloc[0]["name"] == "FWD0"


def test_search_matches_second_name_and_ignores_regex() -> None:
    cat = pd.DataFrame(
        {
            "element_id": [1, 2, 3],
            "name": ["B.Fernandes", "Salah", "Palmer"],
            "first_name": ["Bruno", "Mohamed", "Cole"],
            "second_name": ["Fernandes", "Salah", "Palmer"],
            "web_name": ["B.Fernandes", "Salah", "Palmer"],
            "position": ["MID", "MID", "MID"],
            "team": ["MUN", "LIV", "CHE"],
            "cost_m": [12.0, 14.5, 10.5],
            "points": [0, 0, 0],
        }
    )
    bruno = filter_candidates(cat, position=None, exclude_ids=set(), name="fernandes")
    assert set(bruno["name"]) == {"B.Fernandes"}
    dots = filter_candidates(cat, position=None, exclude_ids=set(), name="B.Fernandes")
    assert set(dots["name"]) == {"B.Fernandes"}
    paren = filter_candidates(cat, position=None, exclude_ids=set(), name="(")
    assert paren.empty


def test_search_across_positions() -> None:
    cat = _catalog()
    out = filter_candidates(cat, position=None, exclude_ids=set(), name="FWD0")
    assert list(out["position"]) == ["FWD"]


def test_exclude_already_picked() -> None:
    cat = _catalog()
    eid = int(cat.loc[cat["position"] == "DEF", "element_id"].iloc[0])
    out = filter_candidates(cat, position="DEF", exclude_ids={eid})
    assert eid not in set(out["element_id"])


def test_min_points_filter() -> None:
    cat = _catalog()
    out = filter_candidates(cat, position="MID", exclude_ids=set(), min_points=20)
    assert (out["points"] >= 20).all()


def test_club_cap_blocks_fourth_from_same_team() -> None:
    cat = pd.DataFrame(
        {
            "element_id": [1, 2, 3, 4],
            "name": ["A", "B", "C", "D"],
            "position": ["MID", "MID", "MID", "MID"],
            "team": ["ARS"] * 4,
            "team_id": [1, 1, 1, 1],
            "cost_m": [5.0, 5.0, 5.0, 5.0],
            "points": [10, 10, 10, 10],
        }
    )
    slots = empty_slots()
    slots["MID"] = [1, 2, 3, None, None]
    assert would_break_club_cap(slots, cat, 4) is True
    assert would_break_club_cap(slots, cat, 4, replacing=3) is False


def test_spent_and_flatten() -> None:
    cat = _catalog()
    slots = fill_from_ids(cat.loc[cat["position"] == "GKP", "element_id"].head(2).tolist(), cat)
    assert flatten_slots(slots) == slots["GKP"]
    assert spent_m(slots, cat) == round(float(cat.loc[cat["element_id"].isin(slots["GKP"]), "cost_m"].sum()), 1)
