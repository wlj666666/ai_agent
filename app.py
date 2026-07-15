"""Streamlit entry point.

Run with:

    streamlit run app.py

All business logic and rendering live in ``drivetest_agent.ui``; this file
only makes sure ``src/`` is importable (independent of the editable
install and of the process's current working directory) and delegates to
``drivetest_agent.ui.page.render_page``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drivetest_agent.ui.page import render_page  # noqa: E402

render_page()
