"""Grounded advisor replies. Optional OpenAI wording; numbers always come from the briefing."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SYSTEM = (
    "You are the FPL analyser's manager. Use ONLY the JSON facts. "
    "Do not invent injuries, lineups, cup form, or news. "
    "If it is not in the facts, say you do not know. "
    "Transfers: recommend TAKE only if expected_net vs hold is positive after 4-pt hits. "
    "Chips are this-GW EV only — never tell the user to auto-play TC/BB/FH/WC. "
    "Be concise. Speak in first person as the engine ('I would hold because...')."
)


def _norm(text: str) -> str:
    return text.strip().lower()


def local_reply(facts: dict[str, Any], question: str) -> str:
    """Answer from briefing facts without a remote model."""
    q = _norm(question)
    recap = facts.get("recap") or {}
    upcoming = facts.get("upcoming") or {}
    flags = facts.get("flags") or []
    players = list(recap.get("players") or [])
    xi = list(upcoming.get("xi") or [])
    names = [p.get("name", "") for p in players + xi + flags]
    mentioned = [n for n in names if n and n.lower() in q]

    if any(w in q for w in ("last", "previous", "recap", "how did", "gw38", "yesterday")):
        bits = [recap.get("headline") or "No last-GW recap.", recap.get("note") or ""]
        if recap.get("best"):
            b = recap["best"]
            bits.append(f"Best: {b['name']} {b['points']:.0f} pts.")
        if recap.get("did_not_play"):
            bits.append("Did not play: " + ", ".join(recap["did_not_play"]) + ".")
        return "\n".join(b for b in bits if b)

    if "captain" in q or "armband" in q:
        cap = upcoming.get("captain_hold") or {}
        return (
            f"I would captain {cap.get('name', '?')} "
            f"(xPts {cap.get('xpts', 0):.2f}, p_play {cap.get('p_play', 0):.2f}). "
            "That is the highest unconditional xPts among players who clear the 0.75 minutes gate."
        )

    if any(w in q for w in ("chip", "triple", "bench boost", "wildcard", "free hit", "bb", "tc")):
        if upcoming.get("chip_beats_no_chip"):
            return (
                f"{upcoming.get('chip_best_label')} wins this week's arithmetic. "
                f"{upcoming.get('chip_note')}"
            )
        return (
            f"I would not play a chip. Best no-chip line is {upcoming.get('best_no_chip')}. "
            f"{upcoming.get('chip_note')}"
        )

    if any(w in q for w in ("transfer", "hit", "hold", "what should", "do this week")):
        if upcoming.get("action") == "TAKE TRANSFERS":
            outs = ", ".join(str(r.get("name")) for r in upcoming.get("transfers_out") or [])
            ins = ", ".join(str(r.get("name")) for r in upcoming.get("transfers_in") or [])
            return (
                f"TAKE: {outs} → {ins}. Expected net vs hold {upcoming.get('expected_net', 0):+.2f} "
                f"after {upcoming.get('hits', 0)} hit(s). Hold EV {upcoming.get('hold_ev', 0):.2f}."
            )
        return (
            f"HOLD. Expected net vs hold is {upcoming.get('expected_net', 0):+.2f} after hits — "
            "not enough to move. Presser minutes go in xmins.csv; I don't scrape news."
        )

    if mentioned:
        lines = []
        for name in mentioned:
            for p in players:
                if p.get("name") == name:
                    lines.append(
                        f"{name} last recap: {p['points']:.0f} pts in {p['minutes']:.0f}' "
                        f"({p['position']})."
                    )
            for p in xi:
                if p.get("name") == name:
                    cap = " Captain." if p.get("captain") else ""
                    lines.append(
                        f"{name} this GW: xPts {p['xpts']:.2f}, p_play {p['p_play']:.2f}.{cap}"
                    )
            for p in flags:
                if p.get("name") == name:
                    lines.append(f"{name} FPL flag: {p.get('status')} — {p.get('news') or 'no news text'}.")
        if lines:
            return "\n".join(lines)

    return (
        "Ask me about last GW, this week's transfers, captain, chips, or a player on your 15. "
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
    }
