"""Shared fixtures for retrieval tests."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


@pytest.fixture
def knowledge_dir() -> Path:
    return KNOWLEDGE_DIR


@pytest.fixture
def sample_markdown_dir(tmp_path: Path) -> Path:
    doc = tmp_path / "sample-spec.md"
    doc.write_text(
        "# 边界测试规范\n\n"
        "阈值测试必须覆盖等于阈值、略低于阈值和略高于阈值三种情况。\n\n"
        "## TTC 边界\n\n"
        "当 TTC 小于等于 1.5 秒时，应验证触发行为；1.49 与 1.51 均需覆盖。\n\n"
        "## 异常输入\n\n"
        "负值、NaN 和非数值输入必须单独设计异常用例。\n",
        encoding="utf-8",
    )
    return tmp_path
