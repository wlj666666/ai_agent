"""Business wiring: safe handling of missing LLM configuration and unexpected errors.

These tests use a fake agent double and never perform a real network call.
"""

from __future__ import annotations

import logging

import pytest

from drivetest_agent.agent.orchestrator import DriveTestAgent
from drivetest_agent.domain.models import AgentReport, Requirement
from drivetest_agent.ui.service import build_agent, missing_llm_config_message, run_requirement

_REQUIREMENT = Requirement(
    text="AEB 模块新增：当 TTC 小于等于 1.5 秒且相对速度为正时触发制动。",
    component_description="模拟 AEB 模块",
)


class _SpyAgent:
    def __init__(
        self, *, report: AgentReport | None = None, exception: Exception | None = None
    ) -> None:
        self._report = report
        self._exception = exception
        self.calls: list[Requirement] = []

    def run(self, requirement: Requirement) -> AgentReport:
        self.calls.append(requirement)
        if self._exception is not None:
            raise self._exception
        assert self._report is not None
        return self._report


def _sample_report() -> AgentReport:
    return AgentReport(
        requirement=_REQUIREMENT,
        knowledge_references=[],
        test_plan=None,
        execution_history=[],
        correction_count=0,
        total_tokens=0,
        total_duration_seconds=0.0,
        final_status="insufficient_info",
        final_test_code=None,
        pass_rate_first_run=None,
        pass_rate_after_correction=None,
        summary="检索到的规范信息不足或置信度过低，未生成测试代码。",
        error_message=None,
    )


class TestMissingLlmConfigMessage:
    def test_returns_message_when_api_key_env_var_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        message = missing_llm_config_message()

        assert message is not None
        assert isinstance(message, str)
        assert message.strip()

    def test_returns_message_when_api_key_env_var_is_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "   ")

        assert missing_llm_config_message() is not None

    def test_returns_none_when_api_key_env_var_is_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        assert missing_llm_config_message() is None


class TestRunRequirementWithMissingConfig:
    def test_returns_safe_error_without_invoking_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = _SpyAgent(report=_sample_report())
        requirement_copy = _REQUIREMENT.model_copy(deep=True)

        outcome = run_requirement(agent, _REQUIREMENT)

        assert outcome.report is None
        assert outcome.error_message is not None
        assert outcome.error_message.strip()
        assert agent.calls == []
        assert _REQUIREMENT == requirement_copy


class TestRunRequirementWithValidConfig:
    def test_delegates_to_agent_and_returns_its_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        report = _sample_report()
        agent = _SpyAgent(report=report)

        outcome = run_requirement(agent, _REQUIREMENT)

        assert outcome.report is report
        assert outcome.error_message is None
        assert agent.calls == [_REQUIREMENT]

    def test_does_not_mutate_requirement_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        agent = _SpyAgent(report=_sample_report())
        requirement_copy = _REQUIREMENT.model_copy(deep=True)

        run_requirement(agent, _REQUIREMENT)

        assert _REQUIREMENT == requirement_copy


class TestRunRequirementUnexpectedException:
    def test_returns_safe_error_and_does_not_leak_exception_detail(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        secret_detail = "internal-secret-token-QRS789"
        agent = _SpyAgent(exception=RuntimeError(f"boom: {secret_detail}"))
        requirement_copy = _REQUIREMENT.model_copy(deep=True)

        with caplog.at_level(logging.ERROR, logger="drivetest_agent.ui.service"):
            outcome = run_requirement(agent, _REQUIREMENT)

        assert outcome.report is None
        assert outcome.error_message is not None
        assert secret_detail not in outcome.error_message
        assert _REQUIREMENT == requirement_copy
        assert "RuntimeError" in caplog.text


class TestBuildAgent:
    def test_returns_a_real_drivetest_agent_instance_without_network_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        agent = build_agent()

        assert isinstance(agent, DriveTestAgent)
