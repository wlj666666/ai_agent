"""Business wiring: safe handling of missing LLM configuration and unexpected errors.

These tests use a fake agent double and never perform a real network call.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from drivetest_agent.agent.orchestrator import DriveTestAgent
from drivetest_agent.config import ConfigError
from drivetest_agent.domain.models import AgentReport, Requirement
from drivetest_agent.ui import service
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

    def test_retriever_uses_default_min_relevance_when_env_var_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RETRIEVAL_MIN_RELEVANCE", raising=False)

        agent = build_agent()

        assert agent._retriever._low_confidence_threshold == pytest.approx(0.15)

    def test_retriever_uses_parsed_min_relevance_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_MIN_RELEVANCE", "0.42")

        agent = build_agent()

        assert agent._retriever._low_confidence_threshold == pytest.approx(0.42)

    def test_raises_config_error_for_non_numeric_min_relevance_without_calling_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("RETRIEVAL_MIN_RELEVANCE", "not-a-number")

        with pytest.raises(ConfigError, match="RETRIEVAL_MIN_RELEVANCE"):
            build_agent()

    def test_raises_config_error_for_out_of_range_min_relevance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_MIN_RELEVANCE", "1.5")

        with pytest.raises(ConfigError, match="RETRIEVAL_MIN_RELEVANCE"):
            build_agent()

    def test_passes_valid_request_timeout_to_openai_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, float] = {}

        class RecordingOpenAIClient:
            def __init__(self, *, timeout_seconds: float) -> None:
                captured["timeout_seconds"] = timeout_seconds

        monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "12.5")
        monkeypatch.setattr(service, "OpenAICompatibleClient", RecordingOpenAIClient)

        agent = build_agent()

        assert captured == {"timeout_seconds": 12.5}
        assert agent._llm_client.__class__ is RecordingOpenAIClient

    def test_invalid_request_timeout_fails_before_openai_client_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class MustNotConstructOpenAIClient:
            def __init__(self, **_kwargs: object) -> None:
                raise AssertionError("OpenAI client must not be constructed")

        monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "nan")
        monkeypatch.setattr(service, "OpenAICompatibleClient", MustNotConstructOpenAIClient)

        with pytest.raises(ConfigError, match="OPENAI_REQUEST_TIMEOUT_SECONDS"):
            build_agent()


class TestDotenvLoading:
    """Dotenv loading always targets a monkeypatched path in tests, never the
    developer's real project ``.env`` (see ``tests/ui/conftest.py``)."""

    def test_missing_llm_config_message_picks_up_key_from_dotenv_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        env_file = tmp_path / "custom.env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
        monkeypatch.setattr(service, "_ENV_PATH", env_file)

        assert missing_llm_config_message() is None
        assert __import__("os").environ["OPENAI_API_KEY"] == "sk-from-dotenv"

    def test_dotenv_file_does_not_override_existing_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-environment")
        env_file = tmp_path / "custom.env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
        monkeypatch.setattr(service, "_ENV_PATH", env_file)

        missing_llm_config_message()

        assert __import__("os").environ["OPENAI_API_KEY"] == "sk-real-environment"

    def test_build_agent_loads_retrieval_min_relevance_from_dotenv_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RETRIEVAL_MIN_RELEVANCE", raising=False)
        env_file = tmp_path / "custom.env"
        env_file.write_text("RETRIEVAL_MIN_RELEVANCE=0.33\n", encoding="utf-8")
        monkeypatch.setattr(service, "_ENV_PATH", env_file)

        agent = build_agent()

        assert agent._retriever._low_confidence_threshold == pytest.approx(0.33)

    def test_missing_dotenv_file_is_a_safe_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(service, "_ENV_PATH", tmp_path / "does-not-exist.env")

        assert missing_llm_config_message() is not None
