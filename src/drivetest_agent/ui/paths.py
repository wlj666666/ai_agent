"""Project-root-relative paths, resolved from this file's own location.

All paths are derived from ``__file__`` rather than the process working
directory, so they resolve correctly no matter where ``streamlit run`` or
``pytest`` is invoked from.
"""

from __future__ import annotations

from pathlib import Path

_PARENTS_FROM_UI_MODULE_TO_PROJECT_ROOT = 3


def compute_project_root(module_file: Path) -> Path:
    """Return the project root given the path to this ``ui/paths.py`` file.

    The layout is fixed: ``<root>/src/drivetest_agent/ui/paths.py``, so the
    project root is always three parents above the resolved module file.
    """
    return module_file.resolve().parents[_PARENTS_FROM_UI_MODULE_TO_PROJECT_ROOT]


PROJECT_ROOT = compute_project_root(Path(__file__))
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
EXAMPLES_PATH = PROJECT_ROOT / "examples" / "cases.json"
