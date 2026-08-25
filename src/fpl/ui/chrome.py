"""Dark My Team chrome for the local Streamlit app. Not a clone of any commercial FPL site."""

from __future__ import annotations

import streamlit as st

CHROME_CSS = """
<style>
header[data-testid="stHeader"] {
  background: #0b0b10;
  border-bottom: 1px solid #2a2933;
}
.stAppDeployButton, div[data-testid="stToolbar"] {
  display: none;
}
footer { visibility: hidden; }
.block-container {
  padding-top: 1.15rem;
  max-width: 1400px;
}
.fpl-topbar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin: 0 0 0.85rem 0;
  padding: 0.15rem 0 0.7rem 0;
  border-bottom: 1px solid #2a2933;
}
.fpl-brand {
  font-size: 1.55rem;
  font-weight: 750;
  letter-spacing: -0.03em;
  color: #f4f4f5;
}
.fpl-topmeta {
  color: #a1a1aa;
  font-size: 0.92rem;
}
.fpl-pill {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.fpl-pill-ok { background: #14532d; color: #bbf7d0; }
.fpl-pill-warn { background: #3f2a00; color: #fde68a; }
.fpl-pill-mute { background: #27272a; color: #d4d4d8; }
.fpl-verdict {
  font-size: 1.7rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  margin: 0.2rem 0 0.4rem 0;
}
.fpl-verdict-take { color: #00ff87; }
.fpl-verdict-hold { color: #e4e4e7; }
</style>
"""


def inject_chrome() -> None:
    st.markdown(CHROME_CSS, unsafe_allow_html=True)


def render_topbar(*, season: str, event: int, status_html: str) -> None:
    st.markdown(
        f'<div class="fpl-topbar">'
        f'<div class="fpl-brand">FPL analyser</div>'
        f'<div class="fpl-topmeta">GW{int(event)} · {season} · {status_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def status_pill(kind: str, text: str) -> str:
    klass = {"ok": "fpl-pill-ok", "warn": "fpl-pill-warn", "mute": "fpl-pill-mute"}[kind]
    return f'<span class="fpl-pill {klass}">{text}</span>'
