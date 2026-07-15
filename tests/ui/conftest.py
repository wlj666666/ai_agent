"""Shared fixtures for UI-layer tests.

Every test under ``tests/ui`` is isolated from the developer's real project
``.env`` file and configuration environment variables: the autouse fixture
always points the service layer's dotenv path at a nonexistent file inside
``tmp_path`` and clears inherited values unless a test explicitly sets them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drivetest_agent.ui import service


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_ENV_PATH", tmp_path / ".env")
    for variable in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_REQUEST_TIMEOUT_SECONDS",
        "RETRIEVAL_MIN_RELEVANCE",
    ):
        monkeypatch.delenv(variable, raising=False)
