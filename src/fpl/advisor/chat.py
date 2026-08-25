"""Grounded advisor replies. Optional OpenAI wording; numbers always come from the briefing."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SYSTEM = (
    "You are a mate in the pub arguing FPL, not a spreadsheet. Use ONLY the JSON facts. "
    "Do not invent injuries, lineups, cup form, or news. If it is not in the facts, say you do not know. "
    "Never say xPts, p_play, FDR, EV, or other short codes. Say expected points this week, "
    "chance he plays, FPL's fixture difficulty (1 easy, 5 brutal), and so on. "
    "When asked why a transfer, argue the full case: expected points, whether they start, price, "
    "official FPL form, last match, official FPL flags, and the next five fixtures. "
    "Say if the fixtures actually support selling him, or if you are moving him for this week only. "
    "Recommend the transfers only if expected_net vs hold is positive after 4-point hits. "
    "Chips are this-week arithmetic only — never tell the user to auto-play TC/BB/FH/WC. "
    "Write like you are talking. Speak in first person."
)


def _fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ı", "i").replace("İ", "i").casefold()


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) if row.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _pair_moves(
    outs: list[dict[str, Any]], ins: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    remaining = list(ins)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for out in outs:
        pos = str(out.get("position") or "")
        idx = next((i for i, inn in enumerate(remaining) if str(inn.get("position") or "") == pos), None)
        if idx is None:
            idx = 0 if remaining else None
        inn = remaining.pop(idx) if idx is not None else {}
        pairs.append((out, inn))
    for inn in remaining:
        pairs.append(({}, inn))
    return pairs


_FIXTURE_RE = re.compile(r"GW(\d+)\s+(\S+)\s+\(([HA])\)\s+FDR(\d|\?)", re.I)


def _flag_sentence(flags: list[dict[str, Any]], name: str) -> str:
    folded = _fold(name)
    for row in flags:
        if _fold(str(row.get("name") or "")) != folded:
            continue
        news = str(row.get("news") or "").strip()
        chance = row.get("chance_of_playing_next_round")
        bits = [f"FPL have flagged {name}"]
        if news:
            bits.append(f": {news.rstrip('.')}.")
        elif not isinstance(chance, (int, float)):
            bits.append(".")
        if isinstance(chance, (int, float)):
            bits.append(f" They only give him a {chance:.0f}% chance of playing next round.")
        bits.append(" That's their official board, not a tweet.")
        return "".join(bits)
    return ""


def _fmt_num(row: dict[str, Any], key: str, digits: int = 1) -> str:
    if row.get(key) is None or str(row.get(key)) == "":
        return ""
    try:
        return f"{float(row[key]):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _club_price(row: dict[str, Any]) -> str:
    club = str(row.get("team") or "").strip()
    cost = row.get("cost_m")
    bits = [b for b in (club,) if b]
    if cost is not None and str(cost) != "":
        bits.append(f"£{float(cost):.1f}m")
    return ", ".join(bits)


def _starts_sentence(row: dict[str, Any], name: str) -> str:
    p = _num(row, "p_play")
    if p >= 0.95:
        return f"{name} should start."
    if p >= 0.75:
        return f"{name} is likely to play, but I would not call him nailed — about a {p:.0%} chance."
    if p >= 0.40:
        return f"{name} is a rotation risk. I only give him about a {p:.0%} chance of featuring."
    return f"I would not bank on {name} playing at all (about {p:.0%})."


def _difficulty_aside(rating: str) -> str:
    if rating == "5":
        return "as hard as FPL will rate a game"
    if rating == "4":
        return "a tough one"
    if rating == "3":
        return "neither kind nor brutal"
    if rating in {"1", "2"}:
        return "a kind one"
    return ""


def _describe_fixture_label(label: str) -> str:
    match = _FIXTURE_RE.search(str(label or ""))
    if not match:
        return str(label or "").strip()
    gw, opp, side, rating = match.group(1), match.group(2), match.group(3), match.group(4)
    where = f"at home to {opp}" if side == "H" else f"away at {opp}"
    aside = _difficulty_aside(rating)
    extra = f" ({aside})" if aside else ""
    return f"gameweek {gw} {where}{extra}"


def _fixtures_story(row: dict[str, Any]) -> str:
    name = str(row.get("name") or "He")
    labels = [p.strip() for p in str(row.get("next_5_text") or "").split(",") if p.strip()]
    if not labels and row.get("this_gw"):
        labels = [str(row["this_gw"])]
    if not labels:
        return ""
    described = [_describe_fixture_label(lab) for lab in labels]
    this = described[0]
    sentences = [f"This week {name} has {this}."]
    if len(described) > 1:
        sentences.append("Then: " + "; ".join(described[1:]) + ".")
    run = _run_kind(row)
    if run == "hard":
        sentences.append("That is a grim stretch.")
    elif run == "kind":
        sentences.append("That run is actually kind.")
    elif run == "mixed":
        sentences.append("Fixtures after that are a mixed bag.")
    return " ".join(sentences)


def _run_kind(row: dict[str, Any]) -> str:
    verdict = str(row.get("fixture_verdict") or "")
    if verdict.startswith("Hard"):
        return "hard"
    if verdict.startswith("Kind"):
        return "kind"
    if verdict.startswith("Mixed"):
        return "mixed"
    next_fdr = row.get("next_fdr")
    try:
        if next_fdr is not None and int(next_fdr) >= 4:
            return "hard"
    except (TypeError, ValueError):
        pass
    return ""


def _fixture_compare(out: dict[str, Any], inn: dict[str, Any]) -> str:
    out_run = _run_kind(out)
    inn_run = _run_kind(inn)
    out_name = str(out.get("name") or "the outgoing player")
    inn_name = str(inn.get("name") or "the incoming player")
    if out_run == "hard" and inn_run != "hard":
        return (
            f"So yes — the fixtures are an argument for selling {out_name}. "
            f"{inn_name} has the kinder run of the two."
        )
    if out_run == "kind" and inn_run == "hard":
        return (
            f"I would not sell {out_name} because of fixtures — his are actually fine, "
            f"and {inn_name}'s look worse. This swap is about this week's expected points, "
            "not a fixture punt."
        )
    if out_run == "kind":
        return (
            f"I would not sell {out_name} because of fixtures — his next five are actually kind. "
            "This swap is about this week's expected points, not a fixture punt."
        )
    if out_run == "hard":
        return f"Those hard games are one reason {out_name} is on the chopping block."
    if out.get("next_5_text") or inn.get("next_5_text"):
        return "Fixtures are mixed, so they are not the main reason for the swap."
    return ""


def _recap_sentence(recap: dict[str, Any], name: str) -> str:
    folded = _fold(name)
    for p in recap.get("players") or []:
        if _fold(str(p.get("name") or "")) == folded:
            return (
                f"Last time out {name} got {p.get('points', 0):.0f} points from "
                f"{p.get('minutes', 0):.0f} minutes."
            )
    return ""


def _price_sentence(out: dict[str, Any], inn: dict[str, Any]) -> str:
    try:
        out_c = float(out["cost_m"]) if out.get("cost_m") is not None else None
        in_c = float(inn["cost_m"]) if inn.get("cost_m") is not None else None
    except (TypeError, ValueError):
        return ""
    if out_c is None or in_c is None:
        return ""
    gap = in_c - out_c
    if gap > 0.05:
        return f"You are not banking cash — {inn.get('name')} costs £{gap:.1f}m more."
    if gap < -0.05:
        return f"It also frees £{-gap:.1f}m, which is a nice side effect, not the reason."
    return ""


def _swap_story(
    out: dict[str, Any],
    inn: dict[str, Any],
    *,
    flags: list[dict[str, Any]],
    recap: dict[str, Any],
    focus: bool = False,
) -> str:
    """Pub-argument paragraph for one out → in. Numbers stay; short codes do not."""
    out_n = str(out.get("name") or "him")
    inn_n = str(inn.get("name") or "the replacement")
    ox = _num(out, "xpts")
    ix = _num(inn, "xpts")
    sentences: list[str] = []
    if focus:
        sentences.append(f"You asked about {out_n}, so I will start there.")
    who_out = _club_price(out)
    who_in = _club_price(inn)
    out_bit = f"{out_n} ({who_out})" if who_out else out_n
    in_bit = f"{inn_n} ({who_in})" if who_in else inn_n
    sentences.append(f"I'd sell {out_bit} for {in_bit}.")
    sentences.append(
        f"This week I only have {out_n} down for about {ox:.2f} points; {inn_n} is more like {ix:.2f}. "
        "That already includes the chance they sit — I am not pretending everyone plays 90."
    )
    out_p = _num(out, "p_play")
    inn_p = _num(inn, "p_play")
    if abs(inn_p - out_p) >= 0.10:
        sentences.append(_starts_sentence(out, out_n) + " " + _starts_sentence(inn, inn_n))
    elif out_p < 0.95 or inn_p < 0.95:
        sentences.append(_starts_sentence(out, out_n) + " " + _starts_sentence(inn, inn_n))
    out_form, inn_form = _fmt_num(out, "form"), _fmt_num(inn, "form")
    if out_form or inn_form:
        sentences.append(
            f"FPL has {out_n}'s recent form at {out_form or '—'} against {inn_n} at {inn_form or '—'}. "
            "That is their last-30-days figure, not something I scraped."
        )
    recap_s = _recap_sentence(recap, out_n)
    if recap_s:
        sentences.append(recap_s)
    fx_out = _fixtures_story(out)
    if fx_out:
        sentences.append(fx_out)
    fx_in = _fixtures_story(inn)
    if fx_in:
        sentences.append(fx_in)
    compare = _fixture_compare(out, inn)
    if compare:
        sentences.append(compare)
    flag = _flag_sentence(flags, out_n)
    if flag:
        sentences.append(flag)
    in_flag = _flag_sentence(flags, inn_n)
    if in_flag:
        sentences.append(in_flag)
    price = _price_sentence(out, inn)
    if price:
        sentences.append(price)
    return " ".join(s for s in sentences if s)


def _hit_sentence(hits: int, n: int) -> str:
    if n <= 0:
        return "I would not move."
    if hits <= 0:
        if n == 1:
            return "One move, and it is a free transfer."
        return f"{n} moves, all on free transfers."
    cost = 4 * hits
    return (
        f"{n} moves, and you would take a {cost}-point hit for the extra "
        f"{'transfer' if hits == 1 else 'transfers'}."
    )


def take_argument(
    upcoming: dict[str, Any],
    recap: dict[str, Any] | None = None,
    flags: list[dict[str, Any]] | None = None,
    *,
    focus: str | None = None,
) -> str:
    """Spoken TAKE (or HOLD) briefing. Same facts, no spreadsheet codes."""
    recap = recap or {}
    flags = flags or []
    outs = list(upcoming.get("transfers_out") or [])
    ins = list(upcoming.get("transfers_in") or [])
    net = _num(upcoming, "expected_net")
    hold_ev = _num(upcoming, "hold_ev")
    chosen_ev = _num(upcoming, "chosen_ev")
    hits = int(upcoming.get("hits") or 0)
    n = int(upcoming.get("n_transfers") or 0)
    action = upcoming.get("action") or "HOLD"
    cap = upcoming.get("captain_hold") or {}

    if action != "TAKE TRANSFERS" or n <= 0:
        return (
            f"**HOLD.** Sitting tight is worth about {hold_ev:.1f} points this week. "
            f"Hunting a transfer only beats that by {net:+.1f} after hits — not enough to move. "
            "Leave it unless a presser changes minutes (`data/overrides/xmins.csv`). "
            "I do not scrape news."
        )

    focus_fold = _fold(focus) if focus else ""
    pairs = _pair_moves(outs, ins)
    if focus_fold:
        pairs = sorted(
            pairs,
            key=lambda pair: 0
            if focus_fold in _fold(f"{pair[0].get('name', '')} {pair[1].get('name', '')}")
            else 1,
        )

    lines = [
        f"**Take them.** {_hit_sentence(hits, n)} "
        f"Doing nothing is worth about {hold_ev:.1f} points. After the hit the new 15 is still "
        f"about {net:.1f} points better ({chosen_ev:.1f} vs {hold_ev:.1f}). "
        "That is the whole case for doing it — not price rises, not your mini-league.",
        "",
    ]
    for out, inn in pairs:
        names = f"{out.get('name', '')} {inn.get('name', '')}"
        is_focus = bool(focus_fold and focus_fold in _fold(names))
        lines.append(_swap_story(out, inn, flags=flags, recap=recap, focus=is_focus))
        lines.append("")
    cap_name = str(cap.get("name") or "?")
    lines.append(
        f"I'd still give the armband to {cap_name}. "
        f"I have him down for about {_num(cap, 'xpts'):.2f} points this week and he looks like starting. "
        "The armband doubles that. I only give it to someone who actually looks like playing — "
        "not the highest ceiling if they happen to start."
    )
    if upcoming.get("chip_beats_no_chip"):
        lines.append(
            f"A chip wins the arithmetic this week ({upcoming.get('chip_best_label')}). "
            "That is not an instruction to play it. "
            f"{upcoming.get('chip_note') or ''}".strip()
        )
    else:
        lines.append(
            "I would not play a chip this week. "
            f"{upcoming.get('chip_note') or ''}".strip()
        )
    lines.append(
        "I am recommending this only because you are still ahead after hits. "
        "It is not team news."
    )
    return "\n".join(line for line in lines if line is not None).strip()


def _named_rows(facts: dict[str, Any]) -> list[dict[str, Any]]:
    upcoming = facts.get("upcoming") or {}
    recap = facts.get("recap") or {}
    rows: list[dict[str, Any]] = []
    for bucket in (
        recap.get("players") or [],
        upcoming.get("xi") or [],
        facts.get("flags") or [],
        upcoming.get("transfers_out") or [],
        upcoming.get("transfers_in") or [],
        [upcoming.get("captain_hold") or {}],
        [upcoming.get("captain_after") or {}],
        facts.get("roster") or [],
    ):
        for row in bucket:
            if row and row.get("name"):
                rows.append(row)
    return rows


def _mentioned(facts: dict[str, Any], question: str) -> list[str]:
    q = _fold(question)
    found: list[str] = []
    for row in _named_rows(facts):
        name = str(row.get("name") or "")
        folded = _fold(name)
        token = folded.split()[-1] if folded else ""
        if len(folded) >= 3 and folded in q:
            found.append(name)
        elif len(token) >= 4 and token in q:
            found.append(name)
    return list(dict.fromkeys(found))


def explain_transfers(facts: dict[str, Any], *, focus: str | None = None) -> str:
    upcoming = facts.get("upcoming") or {}
    return take_argument(
        upcoming,
        facts.get("recap") or {},
        facts.get("flags") or [],
        focus=focus,
    )


def _merged_named(facts: dict[str, Any], name: str) -> dict[str, Any]:
    folded = _fold(name)
    merged: dict[str, Any] = {}
    overlay = {
        "this_gw",
        "next_5_text",
        "next_5_short",
        "fixture_verdict",
        "form",
        "team",
        "cost_m",
        "fdr_mean",
        "hard_n",
        "next_fdr",
    }
    for row in _named_rows(facts):
        if _fold(str(row.get("name") or "")) != folded:
            continue
        for key, value in row.items():
            if value in (None, "", []):
                continue
            if key not in merged or merged[key] in (None, "", []) or key in overlay:
                merged[key] = value
    return merged


def _explain_player(facts: dict[str, Any], name: str, question: str) -> str:
    upcoming = facts.get("upcoming") or {}
    recap = facts.get("recap") or {}
    flags = facts.get("flags") or []
    q = _fold(question)
    out_names = {_fold(str(r.get("name") or "")) for r in upcoming.get("transfers_out") or []}
    in_names = {_fold(str(r.get("name") or "")) for r in upcoming.get("transfers_in") or []}
    folded = _fold(name)
    if folded in out_names or folded in in_names or any(
        _has_word(q, w) for w in ("out", "transfer", "sell", "buy")
    ):
        return explain_transfers(facts, focus=name)

    row = _merged_named(facts, name)
    bits: list[str] = []
    if row:
        who = _club_price(row)
        label = f"{name} ({who})" if who else name
        if row.get("xpts") is not None:
            bits.append(
                f"{label}: I have him down for about {_num(row, 'xpts'):.2f} points this week. "
                + _starts_sentence(row, name)
            )
        else:
            form_s = _fmt_num(row, "form")
            extra = f" FPL has his recent form at {form_s}." if form_s else ""
            bits.append(f"{label}.{extra}".rstrip())
        fx = _fixtures_story(row)
        if fx:
            bits.append(fx)
    recap_s = _recap_sentence(recap, name)
    if recap_s:
        bits.append(recap_s + " Raw combined points — no captain, no auto-subs.")
    for p in upcoming.get("xi") or []:
        if _fold(str(p.get("name") or "")) == folded:
            cap_bit = " He is my captain." if p.get("captain") else ""
            bits.append(
                f"On the side I would field this week, {name} is in the eleven.{cap_bit}"
            )
    flag = _flag_sentence(flags, name)
    if flag:
        bits.append(flag)
    cap = upcoming.get("captain_hold") or {}
    if _fold(str(cap.get("name") or "")) == folded:
        bits.append(
            f"I'd captain {name} because that is the highest expected points "
            "among the players who actually look like starting."
        )
    if not bits:
        bits.append(
            f"{name} is on this briefing but I only have a name — no last-match row and not in the eleven."
        )
    return "\n".join(bits)


def local_reply(facts: dict[str, Any], question: str) -> str:
    """Answer from briefing facts without a remote model."""
    q = _fold(question)
    upcoming = facts.get("upcoming") or {}
    recap = facts.get("recap") or {}
    mentioned = _mentioned(facts, question)

    last_kw = any(_has_word(q, w) for w in ("last", "previous", "recap", "yesterday")) or "how did" in q or "gw38" in q
    if last_kw and not mentioned:
        bits = [recap.get("headline") or "No last-week recap.", recap.get("note") or ""]
        if recap.get("best"):
            b = recap["best"]
            bits.append(f"Best: {b['name']} {b['points']:.0f} points.")
        if recap.get("did_not_play"):
            bits.append("Did not play: " + ", ".join(recap["did_not_play"]) + ".")
        return "\n".join(b for b in bits if b)

    if mentioned:
        return "\n\n".join(_explain_player(facts, name, question) for name in mentioned[:3])

    if _has_word(q, "captain") or _has_word(q, "armband"):
        cap = upcoming.get("captain_hold") or {}
        return (
            f"I'd captain {cap.get('name', '?')}. "
            f"I have him down for about {_num(cap, 'xpts'):.2f} points this week and he looks like starting. "
            "The armband doubles that. I only give it to someone who actually looks like playing — "
            "not the highest ceiling if they happen to start."
        )

    if any(_has_word(q, w) for w in ("chip", "triple", "wildcard")) or "bench boost" in q or "free hit" in q:
        if upcoming.get("chip_beats_no_chip"):
            return (
                f"{upcoming.get('chip_best_label')} wins this week's arithmetic. "
                f"{upcoming.get('chip_note')}"
            )
        return (
            f"I would not play a chip. Best no-chip line is {upcoming.get('best_no_chip')}. "
            f"{upcoming.get('chip_note')}"
        )

    if any(_has_word(q, w) for w in ("transfer", "hit", "hold", "why")) or "what should" in q or "do this week" in q:
        return explain_transfers(facts)

    return (
        "Ask me why a transfer, this week's captain, chips, last GW, or anyone on the move list. "
        "I only use the engine briefing — I will not guess XIs or scrape Twitter."
    )


def advisor_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("FPL_ADVISOR_API_KEY") or None


def llm_reply(facts: dict[str, Any], history: list[dict[str, str]], question: str) -> str:
    key = advisor_api_key()
    if not key:
        return local_reply(facts, question)
    model = os.environ.get("FPL_ADVISOR_MODEL") or "gpt-4o-mini"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "system",
                "content": "Engine facts JSON:\n" + json.dumps(_slim_facts(facts), ensure_ascii=False)[:24000],
            },
            *history[-8:],
            {"role": "user", "content": question},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        os.environ.get("FPL_ADVISOR_URL") or "https://api.openai.com/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return local_reply(facts, question) + f"\n\n(Remote model failed: HTTP {exc.code}; used engine reply.)"
    except URLError as exc:
        return local_reply(facts, question) + f"\n\n(Remote model unreachable: {exc.reason}; used engine reply.)"
    choices = raw.get("choices") or []
    if not choices:
        return local_reply(facts, question)
    content = (choices[0].get("message") or {}).get("content")
    return str(content or local_reply(facts, question))


def _slim_facts(facts: dict[str, Any]) -> dict[str, Any]:
    upcoming = dict(facts.get("upcoming") or {})
    upcoming.pop("transfer_report", None)
    upcoming.pop("chip_report", None)
    recap = dict(facts.get("recap") or {})
    recap["players"] = (recap.get("players") or [])[:15]
    return {
        "recap": recap,
        "upcoming": upcoming,
        "flags": facts.get("flags") or [],
        "roster": (facts.get("roster") or [])[:15],
    }
