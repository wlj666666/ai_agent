"""Single-agent finite-state orchestrator.

Explicit, bounded loop: retrieve -> (generate -> execute) -> optional
one-time correction (generate -> execute) -> report. No hidden framework
loops and no unbounded retries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from drivetest_agent.domain.models import (
    AgentReport,
    AgentState,
    KnowledgeReference,
    Requirement,
    TestExecutionResult,
)
from drivetest_agent.llm.exceptions import LLMFormatError, LLMResponseError, LLMServiceError
from drivetest_agent.llm.prompts import build_correction_prompt, build_generation_prompt
from drivetest_agent.llm.protocol import LLMClient, LLMGeneration
from drivetest_agent.reporting.report_builder import build_report

logger = logging.getLogger(__name__)

_MAX_ERROR_SUMMARY_CHARS_FOR_PROMPT = 1500
_NO_OUTPUT_MESSAGE = "pytest run failed with no captured output."
_TRUNCATION_MARKER = "... [truncated] ...\n"
_SERVICE_ERROR_MESSAGE = "调用模型服务失败，请稍后重试。"
_FORMAT_ERROR_MESSAGE = "模型输出格式在重试后仍不合法，已终止本次生成。"
_RUNNER_ERROR_MESSAGE = "测试执行服务发生异常，已终止本次运行。"

_LLM_ERRORS: tuple[type[Exception], ...] = (LLMServiceError, LLMFormatError, LLMResponseError)


class Retriever(Protocol):
    """Structural contract for the knowledge retriever dependency."""

    def search(self, text: str) -> list[KnowledgeReference]: ...


TestRunner = Callable[[str], TestExecutionResult]


class DriveTestAgent:
    """Runs the fixed retrieve/generate/execute/correct-once workflow."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm_client: LLMClient,
        test_runner: TestRunner,
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._test_runner = test_runner

    def run(self, requirement: Requirement) -> AgentReport:
        state = AgentState(requirement=requirement)

        state.knowledge_references = self._retriever.search(requirement.text)
        if _is_insufficient(state.knowledge_references):
            state.final_status = "insufficient_info"
            return build_report(state)

        try:
            generation = self._llm_client.generate(
                build_generation_prompt(
                    requirement=requirement,
                    knowledge_references=state.knowledge_references,
                )
            )
        except _LLM_ERRORS as exc:
            _log_llm_exception(exc)
            state.final_status = "error"
            return build_report(state, error_message=_error_message_for(exc))

        _apply_generation(state, generation)

        first_result = _run_tests_safely(
            self._test_runner,
            state.current_test_code,  # type: ignore[arg-type]
        )
        if first_result is None:
            state.final_status = "error"
            return build_report(state, error_message=_RUNNER_ERROR_MESSAGE)
        _record_execution(state, first_result)

        if first_result.exit_code == 0:
            state.final_status = "success"
            return build_report(state)

        state.correction_count = 1
        correction_prompt = build_correction_prompt(
            requirement=requirement,
            knowledge_references=state.knowledge_references,
            previous_test_code=state.current_test_code,  # type: ignore[arg-type]
            pytest_error=_truncate_error_summary(first_result.error_summary),
        )

        try:
            correction_generation = self._llm_client.generate(correction_prompt)
        except _LLM_ERRORS as exc:
            _log_llm_exception(exc)
            state.final_status = "error"
            return build_report(state, error_message=_error_message_for(exc))

        _apply_generation(state, correction_generation)

        second_result = _run_tests_safely(
            self._test_runner,
            state.current_test_code,  # type: ignore[arg-type]
        )
        if second_result is None:
            state.final_status = "error"
            return build_report(state, error_message=_RUNNER_ERROR_MESSAGE)
        _record_execution(state, second_result)

        state.final_status = "success" if second_result.exit_code == 0 else "failed"
        return build_report(state)


def _is_insufficient(references: list[KnowledgeReference]) -> bool:
    return not references or all(reference.low_confidence for reference in references)


def _apply_generation(state: AgentState, generation: LLMGeneration) -> None:
    state.test_plan = generation.test_plan
    state.current_test_code = generation.test_plan.pytest_code
    state.total_tokens += generation.tokens
    state.total_duration_seconds += generation.duration_seconds


def _record_execution(state: AgentState, result: TestExecutionResult) -> None:
    state.execution_history.append(result)
    state.total_duration_seconds += result.duration_seconds


def _run_tests_safely(
    test_runner: TestRunner,
    test_code: str,
) -> TestExecutionResult | None:
    try:
        return test_runner(test_code)
    except Exception as exc:
        logger.exception(
            "Test runner failed (%s): %s",
            type(exc).__name__,
            exc,
        )
        return None


def _truncate_error_summary(error_summary: str | None) -> str:
    text = error_summary or _NO_OUTPUT_MESSAGE
    if len(text) <= _MAX_ERROR_SUMMARY_CHARS_FOR_PROMPT:
        return text
    tail_budget = _MAX_ERROR_SUMMARY_CHARS_FOR_PROMPT - len(_TRUNCATION_MARKER)
    if tail_budget <= 0:
        return _TRUNCATION_MARKER[:_MAX_ERROR_SUMMARY_CHARS_FOR_PROMPT]
    return _TRUNCATION_MARKER + text[-tail_budget:]


def _error_message_for(exc: Exception) -> str:
    if isinstance(exc, LLMFormatError):
        return _FORMAT_ERROR_MESSAGE
    return _SERVICE_ERROR_MESSAGE


def _log_llm_exception(exc: Exception) -> None:
    logger.exception(
        "LLM generation failed (%s): %s",
        type(exc).__name__,
        exc,
    )
