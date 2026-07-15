"""End-to-end smoke tests using Streamlit's AppTest harness.

No real API key is configured, so every run exercises the safe
"missing configuration" path only -- no network call is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from drivetest_agent.ui.examples import load_example_cases

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"
_SAMPLE_REQUIREMENT = "AEB 模块新增：当 TTC 小于等于 1.5 秒且相对速度为正时触发制动。"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()
    return at


class TestAppLoads:
    def test_app_runs_without_raising(self, app: AppTest) -> None:
        assert not app.exception

    def test_example_selector_offers_all_fixed_cases(self, app: AppTest) -> None:
        expected_titles = [case.title for case in load_example_cases()]
        assert len(app.selectbox) == 1
        assert list(app.selectbox[0].options) == expected_titles

    def test_requirement_and_component_inputs_start_empty(self, app: AppTest) -> None:
        assert app.text_area(key="requirement_text").value == ""
        assert app.text_area(key="component_description").value == ""

    def test_no_report_or_error_is_shown_before_any_interaction(self, app: AppTest) -> None:
        assert list(app.error) == []
        assert list(app.info)


class TestApplyExampleButton:
    def test_clicking_apply_fills_requirement_text_from_selected_case(self, app: AppTest) -> None:
        first_case = load_example_cases()[0]

        app.button(key="apply_example_button").click().run()

        assert app.text_area(key="requirement_text").value == first_case.requirement
        assert not app.exception


class TestRunWithoutApiKey:
    def test_run_with_empty_requirement_shows_prompt_and_no_report(self, app: AppTest) -> None:
        app.button(key="run_button").click().run()

        assert not app.exception
        assert any("需求" in error.value for error in app.error)

    def test_run_with_text_shows_safe_config_error_and_keeps_input(self, app: AppTest) -> None:
        app.text_area(key="requirement_text").set_value(_SAMPLE_REQUIREMENT).run()
        app.button(key="run_button").click().run()

        assert not app.exception
        assert any("OPENAI_API_KEY" in error.value for error in app.error)
        assert app.text_area(key="requirement_text").value == _SAMPLE_REQUIREMENT
        assert not list(app.code)


class TestRunWithInvalidRetrievalConfig:
    def test_run_shows_specific_config_error_without_generic_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("RETRIEVAL_MIN_RELEVANCE", "not-a-number")
        at = AppTest.from_file(str(APP_PATH), default_timeout=15)
        at.run()
        at.text_area(key="requirement_text").set_value(_SAMPLE_REQUIREMENT).run()

        at.button(key="run_button").click().run()

        assert not at.exception
        assert any("RETRIEVAL_MIN_RELEVANCE" in error.value for error in at.error)
        assert not any("未预期" in error.value for error in at.error)


class TestRunWithInvalidRequestTimeout:
    def test_run_shows_specific_timeout_config_error_without_calling_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "0")
        at = AppTest.from_file(str(APP_PATH), default_timeout=15)
        at.run()
        at.text_area(key="requirement_text").set_value(_SAMPLE_REQUIREMENT).run()

        at.button(key="run_button").click().run()

        assert not at.exception
        assert any("OPENAI_REQUEST_TIMEOUT_SECONDS" in error.value for error in at.error)
        assert not list(at.code)
