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
    "You are the FPL analyser's manager. Use ONLY the JSON facts. "
    "Do not invent injuries, lineups, cup form, or news. "
    "If it is not in the facts, say you do not know. "
    "When asked why a transfer, dump the full sale case: this-GW xPts (already haircut by p_play), "
    "p_play, price, official FPL form, last recap minutes/points, FPL flags, and next-5 FDR "
    "(1 easiest, 5 hardest). Say explicitly if fixtures support the sale or if the swap is "
    "this-GW xPts only. TAKE only if expected_net vs hold is positive after 4-pt hits. "
    "Chips are this-GW EV only — never tell the user to auto-play TC/BB/FH/WC. "
    "Write a short analysis (not a one-liner). Speak in first person as the engine."
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


def _flag_line(flags: list[dict[str, Any]], name: str) -> str:
    folded = _fold(name)
    for row in flags:
        if _fold(str(row.get("name") or "")) == folded:
            news = str(row.get("news") or "").strip()
            chance = row.get("chance_of_playing_next_round")
            extra = f" ({chance:.0f}% next)" if isinstance(chance, (int, float)) else ""
            return f" FPL flag: {row.get('status')}{extra}" + (f" — {news}" if news else ".")
    return ""


def _fmt_num(row: dict[str, Any], key: str, digits: int = 1) -> str:
    if row.get(key) is None or str(row.get(key)) == "":
        return "—"
    try:
        return f"{float(row[key]):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_asset(row: dict[str, Any]) -> str:
    if not row:
        return "—"
    name = str(row.get("name") or "?")
    pos = str(row.get("position") or "")
    club = str(row.get("team") or "").strip()
    cost = row.get("cost_m")
    bits = [b for b in (pos, club) if b]
    if cost is not None and str(cost) != "":
        bits.append(f"£{float(cost):.1f}m")
    prefix = ", ".join(bits)
    form_s = _fmt_num(row, "form", 1)
    return (
        f"{name} ({prefix}: xPts {_num(row, 'xpts'):.2f}, "
        f"p_play {_num(row, 'p_play'):.2f}, FPL form {form_s})"
    )


def _fixture_block(row: dict[str, Any]) -> str:
    if not row:
        return ""
    parts: list[str] = []
    if row.get("this_gw"):
        parts.append(f"this GW {row['this_gw']}")
    if row.get("next_5_text"):
        parts.append(f"next 5 (official FPL FDR, 5=hardest): {row['next_5_text']}")
    if row.get("fixture_verdict"):
        parts.append(str(row["fixture_verdict"]))
    return ". ".join(parts)


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
            f"Fixture run supports selling {out_name}: hard stretch vs "
            f"{inn_name}'s {inn_run or 'easier'} run."
        )
    if out_run == "kind" and inn_run == "hard":
        return (
            f"Fixtures argue against selling {out_name} — kind run vs {inn_name}'s hard stretch. "
            "The swap is this-GW xPts (and minutes/flags), not a fixture punt."
        )
    if out_run == "kind":
        return (
            f"Fixtures do not argue for selling {out_name} — his next 5 are actually kind. "
            "The swap is this-GW xPts, not a fixture punt."
        )
    if out_run == "hard":
        return f"Hard fixtures are one reason {out_name} is on the chopping block."
    if out.get("next_5_text") or inn.get("next_5_text"):
        return "Fixtures are mixed; they are not the main reason for the swap."
    return ""


def _recap_line(recap: dict[str, Any], name: str) -> str:
    folded = _fold(name)
    for p in recap.get("players") or []:
        if _fold(str(p.get("name") or "")) == folded:
            return (
                f"Last recap for {name}: {p.get('points', 0):.0f} pts in {p.get('minutes', 0):.0f}'."
            )
    return ""


def _sale_case(
    out: dict[str, Any],
    inn: dict[str, Any],
    *,
    flags: list[dict[str, Any]],
    recap: dict[str, Any],
) -> list[str]:
    """Every official fact that supports (or does not support) selling `out`."""
    lines: list[str] = []
    delta = _num(inn, "xpts") - _num(out, "xpts")
    lines.append(
        f"  This-GW xPts {_num(out, 'xpts'):.2f} vs {_num(inn, 'xpts'):.2f} "
        f"({delta:+.2f}), already haircut by p_play "
        f"({_num(out, 'p_play'):.2f} vs {_num(inn, 'p_play'):.2f})."
    )
    if _num(inn, "p_play") - _num(out, "p_play") >= 0.10:
        lines.append(
            f"  Minutes: {out.get('name')} is less likely to play than {inn.get('name')}."
        )
    out_form, inn_form = out.get("form"), inn.get("form")
    if isinstance(out_form, (int, float)) or isinstance(inn_form, (int, float)):
        lines.append(
            f"  Official FPL form {_fmt_num(out, 'form')} vs {_fmt_num(inn, 'form')} "
            "(FPL's last-30-days figure, not a scrape)."
        )
    recap_line = _recap_line(recap, str(out.get("name") or ""))
    if recap_line:
        lines.append(f"  {recap_line}")
    out_fx = _fixture_block(out)
    if out_fx:
        lines.append(f"  {out.get('name')}: {out_fx}")
    inn_fx = _fixture_block(inn)
    if inn_fx:
        lines.append(f"  {inn.get('name')}: {inn_fx}")
    compare = _fixture_compare(out, inn)
    if compare:
        lines.append(f"  {compare}")
    flag = _flag_line(flags, str(out.get("name") or "")).strip()
    if flag:
        lines.append(f"  {out.get('name')} {flag}")
    in_flag = _flag_line(flags, str(inn.get("name") or "")).strip()
    if in_flag:
        lines.append(f"  {inn.get('name')} {in_flag}")
    try:
        out_c = float(out["cost_m"]) if out.get("cost_m") is not None else None
        in_c = float(inn["cost_m"]) if inn.get("cost_m") is not None else None
    except (TypeError, ValueError):
        out_c, in_c = None, None
    if out_c is not None and in_c is not None:
        gap = in_c - out_c
        if gap > 0.05:
            lines.append(f"  Price: not a cash-out — you spend £{gap:.1f}m more.")
        elif gap < -0.05:
            lines.append(f"  Price: frees £{-gap:.1f}m.")
    return lines


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
    flags = facts.get("flags") or []
    recap = facts.get("recap") or {}
    outs = list(upcoming.get("transfers_out") or [])
    ins = list(upcoming.get("transfers_in") or [])
    net = _num(upcoming, "expected_net")
    hold_ev = _num(upcoming, "hold_ev")
    chosen_ev = _num(upcoming, "chosen_ev")
    hits = int(upcoming.get("hits") or 0)
    n = int(upcoming.get("n_transfers") or 0)
    action = upcoming.get("action") or "HOLD"

    if action != "TAKE TRANSFERS" or n <= 0:
        return (
            f"HOLD. The transfer search is {net:+.2f} expected points vs doing nothing "
            f"(hold EV {hold_ev:.2f}) after 4-point hits. That is not enough to move. "
            "If a presser changes minutes, put it in `data/overrides/xmins.csv` — I do not scrape news."
        )

    hit_txt = f"{hits} hit(s) (−{4 * hits} pts)" if hits else "no hit (free transfer(s))"
    lines = [
        f"TAKE: {n} move(s), {hit_txt}. "
        f"Hold EV {hold_ev:.2f} → chosen net {chosen_ev:.2f} ({net:+.2f} vs hold after hits).",
        "",
        "Why each move — every official fact I have, not news:",
    ]
    focus_fold = _fold(focus) if focus else ""
    for out, inn in _pair_moves(outs, ins):
        delta = _num(inn, "xpts") - _num(out, "xpts")
        mark = ""
        names = f"{out.get('name', '')} {inn.get('name', '')}"
        if focus_fold and focus_fold in _fold(names):
            mark = " ← this is the one you asked about"
        lines.append(f"- {_fmt_asset(out)} → {_fmt_asset(inn)}. Swap {delta:+.2f} xPts.{mark}")
        lines.extend(_sale_case(out, inn, flags=flags, recap=recap))
    cap = upcoming.get("captain_hold") or {}
    lines.append(
        f"Captain on the hold 15 is {cap.get('name', '?')} "
        f"(xPts {_num(cap, 'xpts'):.2f}) unless the new 15 changes the armband."
    )
    lines.append(
        "I recommend TAKE only because that net is positive after hits. "
        "This is not a price-rise or mini-league call, and it is not team news."
    )
    if upcoming.get("engine_note"):
        lines.append(str(upcoming["engine_note"]))
    return "\n".join(lines)


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
        if row.get("xpts") is not None:
            bits.append(_fmt_asset(row))
        else:
            club = str(row.get("team") or "").strip()
            cost = row.get("cost_m")
            cost_s = f" £{float(cost):.1f}m" if cost is not None and str(cost) != "" else ""
            form_s = _fmt_num(row, "form")
            bits.append(f"{name} ({club}{cost_s}, FPL form {form_s})".replace("( ", "("))
        fx = _fixture_block(row)
        if fx:
            bits.append(fx)
    recap_line = _recap_line(recap, name)
    if recap_line:
        bits.append(recap_line + " Raw combined points — no captain, no auto-subs.")
    for p in upcoming.get("xi") or []:
        if _fold(str(p.get("name") or "")) == folded:
            cap = " Captain." if p.get("captain") else ""
            bits.append(
                f"{name} this GW on the hold XI: xPts {_num(p, 'xpts'):.2f}, "
                f"p_play {_num(p, 'p_play'):.2f}.{cap}"
            )
    flag = _flag_line(flags, name).strip()
    if flag:
        bits.append(f"{name} {flag}")
    cap = upcoming.get("captain_hold") or {}
    if _fold(str(cap.get("name") or "")) == folded:
        bits.append(
            f"I would captain {name} because that is the highest unconditional xPts "
            "among players with p_play ≥ 0.75."
        )
    if not bits:
        bits.append(
            f"{name} is on this briefing but I only have a name — no recap row and not in the hold XI."
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
        bits = [recap.get("headline") or "No last-GW recap.", recap.get("note") or ""]
        if recap.get("best"):
            b = recap["best"]
            bits.append(f"Best: {b['name']} {b['points']:.0f} pts.")
        if recap.get("did_not_play"):
            bits.append("Did not play: " + ", ".join(recap["did_not_play"]) + ".")
        return "\n".join(b for b in bits if b)

    if mentioned:
        return "\n\n".join(_explain_player(facts, name, question) for name in mentioned[:3])

    if _has_word(q, "captain") or _has_word(q, "armband"):
        cap = upcoming.get("captain_hold") or {}
        return (
            f"I would captain {cap.get('name', '?')} "
            f"(xPts {_num(cap, 'xpts'):.2f}, p_play {_num(cap, 'p_play'):.2f}). "
            "Armband is 2× unconditional xPts among players who clear the 0.75 minutes gate — "
            "not the highest if-he-plays ceiling, and not a vibe pick."
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
