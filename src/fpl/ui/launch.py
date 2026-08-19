"""Launch the local Streamlit UI from the project root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fpl.paths import project_root


def launch() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. From the venv run:  pip install streamlit",
            file=sys.stderr,
        )
        return 1
    app = Path(__file__).resolve().parent / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--browser.gatherUsageStats=false",
    ]
    return int(subprocess.call(cmd, cwd=project_root()))
