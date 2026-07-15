"""Integration tests for the single-agent finite-state orchestrator."""

from __future__ import annotations

import logging
from collections import deque

import pytest

from drivetest_agent.agent.orchestrator import DriveTestAgent
from drivetest_agent.domain.models import (
    KnowledgeReference,
    Requirement,
    TestCasePlan,
    TestExecutionResult,
    TestPlan,
)
from drivetest_agent.llm.exceptions import LLMFormatError, LLMServiceError
from drivetest_agent.llm.fake_client import FakeLLMClient
from drivetest_agent.llm.protocol import LLMGeneration


class StubRetriever:
    """Returns a fixed list of references and records queries."""

    def __init__(self, references: list[KnowledgeReference]) -> None:
        self._references = references
        self.queries: list[str] = []

    def search(self, text: str) -> list[KnowledgeReference]:
        self.queries.append(text)
        return self._references


class FakeTestRunner:
    """Returns queued execution results and records the code it was given."""

    def __init__(self, results: list[TestExecutionResult | Exception]) -> None:
        self._results: deque[TestExecutionResult | Exception] = deque(results)
        self.received_code: list[str] = []

    def __call__(self, test_code: str) -> TestExecutionResult:
        self.received_code.append(test_code)
        if not self._results:
            raise AssertionError("test runner invoked more times than expected")
        item = self._results.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def _high_confidence_references() -> list[KnowledgeReference]:
    return [
        KnowledgeReference(
            source="knowledge/aeb-input-constraints.md",
            snippet="TTC must be non-negative and finite.",
            relevance_score=0.8,
            low_confidence=False,
        ),
        KnowledgeReference(
            source="knowledge/boundary-exception-testing.md",
            snippet="Cover both sides of numeric thresholds.",
            relevance_score=0.7,
            low_confidence=False,
        ),
    ]


def _low_confidence_references() -> list[KnowledgeReference]:
    return [
        KnowledgeReference(
            source="knowledge/aeb-input-constraints.md",
            snippet="Weak match only.",
            relevance_score=0.02,
            low_confidence=True,
        ),
    ]


def _test_plan(name: str) -> TestPlan:
    return TestPlan(
        test_cases=[
            TestCasePlan(
                name=name,
                description="Verify braking behaviour.",
                expected_outcome="Braking is triggered as expected.",
            )
        ],
        pytest_code=f"def {name}():\n    assert True\n",
    )


def _generation(name: str, *, tokens: int, duration_seconds: float) -> LLMGeneration:
    return LLMGeneration(
        test_plan=_test_plan(name),
        tokens=tokens,
        duration_seconds=duration_seconds,
    )


def _passing_result(*, duration_seconds: float = 0.1) -> TestExecutionResult:
    return TestExecutionResult(
        exit_code=0,
        passed=1,
        failed=0,
        duration_seconds=duration_seconds,
        error_summary=None,
        timed_out=False,
    )


def _failing_result(
    *, duration_seconds: float = 0.15, error_summary: str = "AssertionError: expected True"
) -> TestExecutionResult:
    return TestExecutionResult(
        exit_code=1,
        passed=0,
        failed=1,
        duration_seconds=duration_seconds,
        error_summary=error_summary,
        timed_out=False,
    )


class TestDriveTestAgentSuccessPath:
    def test_first_run_passes_without_correction(self, requirement: Requirement) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        test_runner = FakeTestRunner([_passing_result(duration_seconds=0.1)])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "success"
        assert report.correction_count == 0
        assert len(report.execution_history) == 1
        assert report.final_test_code == "def test_first_attempt():\n    assert True\n"
        assert len(llm_client.prompts) == 1
        assert len(test_runner.received_code) == 1
        assert retriever.queries == [requirement.text]


class TestDriveTestAgentCorrectionPath:
    def test_first_run_fails_then_correction_passes(self, requirement: Requirement) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        test_runner = FakeTestRunner(
            [
                _failing_result(duration_seconds=0.15),
                _passing_result(duration_seconds=0.1),
            ]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "success"
        assert report.correction_count == 1
        assert len(report.execution_history) == 2
        assert report.final_test_code == "def test_corrected_attempt():\n    assert True\n"
        assert len(llm_client.prompts) == 2
        assert len(test_runner.received_code) == 2

    def test_correction_prompt_includes_previous_code_and_error(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        test_runner = FakeTestRunner(
            [
                _failing_result(
                    duration_seconds=0.15,
                    error_summary="AssertionError: expected braking to trigger",
                ),
                _passing_result(duration_seconds=0.1),
            ]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        agent.run(requirement)

        correction_prompt = llm_client.prompts[1]
        assert "def test_first_attempt():" in correction_prompt
        assert "AssertionError: expected braking to trigger" in correction_prompt
        assert "=== PREVIOUS_TEST ===" in correction_prompt
        assert "=== PYTEST_ERROR ===" in correction_prompt

    def test_correction_prompt_truncates_very_long_error_summary(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        long_error = "HEAD_MARKER\n" + ("noise line\n" * 500) + "TAIL_MARKER"
        test_runner = FakeTestRunner(
            [
                _failing_result(duration_seconds=0.15, error_summary=long_error),
                _passing_result(duration_seconds=0.1),
            ]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        agent.run(requirement)

        correction_prompt = llm_client.prompts[1]
        assert "HEAD_MARKER" not in correction_prompt
        assert "TAIL_MARKER" in correction_prompt

    def test_timed_out_first_run_enters_single_correction(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        timed_out_result = TestExecutionResult(
            exit_code=-1,
            passed=0,
            failed=0,
            duration_seconds=5.0,
            error_summary="Execution timed out.",
            timed_out=True,
        )
        test_runner = FakeTestRunner([timed_out_result, _passing_result()])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "success"
        assert report.correction_count == 1
        assert report.execution_history[0].timed_out is True
        assert len(llm_client.prompts) == 2
        assert len(test_runner.received_code) == 2


class TestDriveTestAgentFailurePath:
    def test_two_failures_stop_after_exactly_two_attempts(self, requirement: Requirement) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        test_runner = FakeTestRunner(
            [
                _failing_result(duration_seconds=0.15),
                _failing_result(duration_seconds=0.12),
            ]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "failed"
        assert report.correction_count == 1
        assert len(report.execution_history) == 2
        assert len(llm_client.prompts) == 2
        assert len(test_runner.received_code) == 2
        assert report.final_test_code == "def test_corrected_attempt():\n    assert True\n"


class TestDriveTestAgentInsufficientInfo:
    def test_no_references_returns_insufficient_info_without_calling_dependencies(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever([])
        llm_client = FakeLLMClient()
        test_runner = FakeTestRunner([])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "insufficient_info"
        assert report.final_test_code is None
        assert report.correction_count == 0
        assert llm_client.prompts == []
        assert test_runner.received_code == []

    def test_all_low_confidence_references_returns_insufficient_info(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever(_low_confidence_references())
        llm_client = FakeLLMClient()
        test_runner = FakeTestRunner([])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.final_status == "insufficient_info"
        assert report.knowledge_references == _low_confidence_references()
        assert llm_client.prompts == []
        assert test_runner.received_code == []


class TestDriveTestAgentLLMErrors:
    def test_llm_format_error_on_first_generate_returns_error_report(
        self, requirement: Requirement, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        secret_detail = "internal-stack-trace-XYZ123"
        llm_client.enqueue(LLMFormatError(f"schema invalid: {secret_detail}"))
        test_runner = FakeTestRunner([])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        with caplog.at_level(logging.ERROR, logger="drivetest_agent.agent.orchestrator"):
            report = agent.run(requirement)

        assert report.final_status == "error"
        assert report.final_test_code is None
        assert report.correction_count == 0
        assert test_runner.received_code == []
        assert report.error_message
        assert secret_detail not in (report.error_message or "")
        assert secret_detail not in (report.summary or "")
        assert "LLMFormatError" in caplog.text
        assert secret_detail in caplog.text

    def test_llm_service_error_on_correction_generate_preserves_prior_state(
        self, requirement: Requirement, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        secret_detail = "db-connection-string://internal-host:5432"
        llm_client.enqueue(LLMServiceError(f"upstream failure: {secret_detail}"))
        test_runner = FakeTestRunner([_failing_result(duration_seconds=0.15)])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        with caplog.at_level(logging.ERROR, logger="drivetest_agent.agent.orchestrator"):
            report = agent.run(requirement)

        assert report.final_status == "error"
        assert report.correction_count == 1
        assert len(report.execution_history) == 1
        assert len(test_runner.received_code) == 1
        assert secret_detail not in (report.error_message or "")
        assert secret_detail not in (report.summary or "")
        assert "LLMServiceError" in caplog.text
        assert secret_detail in caplog.text


class TestDriveTestAgentRunnerErrors:
    def test_first_runner_exception_returns_safe_error_report_and_logs_diagnostics(
        self, requirement: Requirement, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        secret_detail = "runner-secret-token-ABC123"
        test_runner = FakeTestRunner([RuntimeError(f"runner crashed: {secret_detail}")])

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        with caplog.at_level(logging.ERROR, logger="drivetest_agent.agent.orchestrator"):
            report = agent.run(requirement)

        assert report.final_status == "error"
        assert report.final_test_code == "def test_first_attempt():\n    assert True\n"
        assert report.test_plan == _test_plan("test_first_attempt")
        assert report.execution_history == []
        assert report.correction_count == 0
        assert report.total_tokens == 120
        assert report.total_duration_seconds == pytest.approx(0.3)
        assert secret_detail not in (report.error_message or "")
        assert secret_detail not in (report.summary or "")
        assert "RuntimeError" in caplog.text
        assert secret_detail in caplog.text

    def test_second_runner_exception_preserves_corrected_state_and_metrics(
        self, requirement: Requirement, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        first_result = _failing_result(duration_seconds=0.15)
        secret_detail = "second-runner-secret-XYZ789"
        test_runner = FakeTestRunner(
            [first_result, ValueError(f"runner rejected code: {secret_detail}")]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        with caplog.at_level(logging.ERROR, logger="drivetest_agent.agent.orchestrator"):
            report = agent.run(requirement)

        assert report.final_status == "error"
        assert report.execution_history == [first_result]
        assert report.test_plan == _test_plan("test_corrected_attempt")
        assert report.final_test_code == "def test_corrected_attempt():\n    assert True\n"
        assert report.correction_count == 1
        assert report.total_tokens == 200
        assert report.total_duration_seconds == pytest.approx(0.65)
        assert len(llm_client.prompts) == 2
        assert len(test_runner.received_code) == 2
        assert secret_detail not in (report.error_message or "")
        assert secret_detail not in (report.summary or "")
        assert "ValueError" in caplog.text
        assert secret_detail in caplog.text


class TestDriveTestAgentMetrics:
    def test_accumulates_tokens_and_durations_across_two_llm_and_runner_calls(
        self, requirement: Requirement
    ) -> None:
        retriever = StubRetriever(_high_confidence_references())
        llm_client = FakeLLMClient()
        llm_client.enqueue(_generation("test_first_attempt", tokens=120, duration_seconds=0.3))
        llm_client.enqueue(_generation("test_corrected_attempt", tokens=80, duration_seconds=0.2))
        test_runner = FakeTestRunner(
            [
                _failing_result(duration_seconds=0.15),
                _passing_result(duration_seconds=0.1),
            ]
        )

        agent = DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=test_runner)
        report = agent.run(requirement)

        assert report.total_tokens == 200
        assert report.total_duration_seconds == pytest.approx(0.75)
