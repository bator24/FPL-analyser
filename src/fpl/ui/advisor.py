"""Streamlit advisor: briefing plus grounded chat."""

from __future__ import annotations

import streamlit as st

from fpl.advisor.briefing import build_weekly_briefing, write_briefing
from fpl.advisor.chat import advisor_api_key, llm_reply, local_reply
from fpl.optimize.rules import SQUAD_SIZE


def render_advisor(
    *,
    season: str,
    event: int,
    free_transfers: int,
    squad_ids: list[int],
) -> None:
    st.caption(
        "Generate a last-GW recap plus this week's HOLD/TAKE and captain (same engine as **Transfers**). "
        "Then ask **why that transfer** — the reply dumps the sale case (xPts, form, FDR, flags), not a one-liner. "
        "Full click-order is in **How to use this app** at the top."
    )
    ready = len({int(i) for i in squad_ids}) == SQUAD_SIZE
    if st.button("Generate this week's briefing", type="primary", disabled=not ready):
        with st.spinner("Scoring your 15 and writing the briefing…"):
            try:
                result = build_weekly_briefing(
                    season=str(season),
                    event=int(event),
                    squad_ids=list(squad_ids),
                    free_transfers=int(free_transfers),
                )
                path = write_briefing(result)
                st.session_state.briefing_md = result["markdown"]
                st.session_state.briefing_facts = result["facts"]
                st.session_state.advisor_chat = [
                    {
                        "role": "assistant",
                        "content": (
                            "Briefing is above. Ask me about last GW, this week's transfers, "
                            "captain, chips, or anyone on your 15."
                        ),
                    }
                ]
                st.success(f"Wrote {path}")
            except Exception as exc:
                st.error(str(exc))
    if not ready:
        st.info("Save a legal 15 on the pitch first (or load your FPL team).")

    md = st.session_state.get("briefing_md")
    if md:
        st.markdown(md)

    facts = st.session_state.get("briefing_facts")
    if not facts:
        return

    st.markdown("**Talk it through**")
    remote = bool(advisor_api_key())
    st.caption(
        "Replies stay on this briefing. "
        + (
            "OPENAI_API_KEY is set — wording may go through that API (squad facts leave your machine)."
            if remote
            else "No API key: the engine still explains each move from xPts. Set OPENAI_API_KEY only if you want it rephrased."
        )
    )
    for message in st.session_state.get("advisor_chat") or []:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Why this captain? Should I take the hit? How did X do?")
    if not prompt:
        return
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in (st.session_state.get("advisor_chat") or [])
        if m["role"] in {"user", "assistant"}
    ]
    # Don't resend the full briefing markdown as every history turn if it's huge;
    # facts JSON is in the system prompt for the remote model.
    slim_history = [m for m in history if not m["content"].startswith("# Weekly briefing")][-6:]
    try:
        answer = llm_reply(facts, slim_history, prompt) if remote else local_reply(facts, prompt)
    except Exception as exc:
        answer = f"Advisor failed: {exc}"
    st.session_state.advisor_chat = (st.session_state.get("advisor_chat") or []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    st.rerun()
