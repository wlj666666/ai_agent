"""Loader for the fixed demo cases in examples/cases.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drivetest_agent.ui.examples import ExampleCase, load_example_cases
from drivetest_agent.ui.paths import EXAMPLES_PATH


def _write_cases(tmp_path: Path, payload: object) -> Path:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return cases_path


class TestLoadExampleCasesHappyPath:
    def test_loads_three_valid_cases_into_example_case_models(self, tmp_path: Path) -> None:
        payload = [
            {"id": "normal", "title": "正常需求", "requirement": "需求文本一"},
            {"id": "boundary", "title": "边界需求", "requirement": "需求文本二"},
            {
                "id": "insufficient",
                "title": "信息不足需求",
                "requirement": "需求文本三",
                "component_description": "组件说明",
            },
        ]
        cases_path = _write_cases(tmp_path, payload)

        cases = load_example_cases(cases_path)

        assert len(cases) == 3
        assert all(isinstance(case, ExampleCase) for case in cases)
        assert cases[0].id == "normal"
        assert cases[2].component_description == "组件说明"

    def test_component_description_defaults_to_none_when_absent(self, tmp_path: Path) -> None:
        payload = [{"id": "a", "title": "标题", "requirement": "需求"}]
        cases_path = _write_cases(tmp_path, payload)

        cases = load_example_cases(cases_path)

        assert cases[0].component_description is None


class TestLoadExampleCasesValidation:
    def test_raises_when_id_is_duplicated(self, tmp_path: Path) -> None:
        payload = [
            {"id": "dup", "title": "标题一", "requirement": "需求一"},
            {"id": "dup", "title": "标题二", "requirement": "需求二"},
        ]
        cases_path = _write_cases(tmp_path, payload)

        with pytest.raises(ValueError, match="dup"):
            load_example_cases(cases_path)

    def test_raises_when_id_is_empty(self, tmp_path: Path) -> None:
        payload = [{"id": "", "title": "标题", "requirement": "需求"}]
        cases_path = _write_cases(tmp_path, payload)

        with pytest.raises(ValueError):
            load_example_cases(cases_path)

    def test_raises_when_title_is_blank(self, tmp_path: Path) -> None:
        payload = [{"id": "a", "title": "   ", "requirement": "需求"}]
        cases_path = _write_cases(tmp_path, payload)

        with pytest.raises(ValueError):
            load_example_cases(cases_path)

    def test_raises_when_requirement_is_missing(self, tmp_path: Path) -> None:
        payload = [{"id": "a", "title": "标题"}]
        cases_path = _write_cases(tmp_path, payload)

        with pytest.raises(ValueError):
            load_example_cases(cases_path)

    def test_raises_when_file_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_example_cases(tmp_path / "missing.json")

    def test_raises_when_top_level_json_is_not_a_list(self, tmp_path: Path) -> None:
        cases_path = _write_cases(tmp_path, {"id": "a"})

        with pytest.raises(ValueError):
            load_example_cases(cases_path)


class TestRealFixedCasesFile:
    def test_default_examples_path_loads_exactly_three_unique_cases(self) -> None:
        cases = load_example_cases(EXAMPLES_PATH)

        assert len(cases) == 3
        assert len({case.id for case in cases}) == 3
        for case in cases:
            assert case.title.strip()
            assert case.requirement.strip()

    def test_fixed_cases_cover_normal_boundary_and_insufficient_scenarios(self) -> None:
        cases = load_example_cases(EXAMPLES_PATH)
        combined_text = " ".join(case.requirement for case in cases)

        assert "1.5" in combined_text
        assert "1.49" in combined_text
        assert "1.51" in combined_text

    def test_load_example_cases_without_explicit_path_uses_default(self) -> None:
        cases = load_example_cases()

        assert len(cases) == 3
