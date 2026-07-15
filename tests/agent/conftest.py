"""Shared fixtures for agent orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from drivetest_agent.domain.models import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


@pytest.fixture
def knowledge_dir() -> Path:
    return KNOWLEDGE_DIR


@pytest.fixture
def requirement() -> Requirement:
    return Requirement(
        text="AEB 模块新增：当 TTC 小于等于 1.5 秒且相对速度为正时触发制动。",
        component_description="Simulated AEB module",
    )
