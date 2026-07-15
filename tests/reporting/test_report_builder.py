"""Tests for AgentReport construction and pass-rate computation."""

from __future__ import annotations

import pytest

from drivetest_agent.domain.models import (
    AgentState,
    KnowledgeReference,
    Requirement,
    TestCasePlan,
    TestExecutionResult,
    TestPlan,
)
from drivetest_agent.reporting.report_builder import build_report, compute_pass_rate


def _sample_requirement() -> Requirement:
    return Requirement(text="AEB triggers when TTC <= 1.5s and relative speed is positive.")


def _sample_reference() -> KnowledgeReference:
    return KnowledgeReference(
        source="knowledge/aeb-input-constraints.md",
        snippet="TTC must be non-negative and finite.",
        relevance_score=0.8,
    )


def _sample_test_plan() -> TestPlan:
    return TestPlan(
        test_cases=[
            TestCasePlan(
                name="test_threshold",
                description="Verify braking at threshold.",
                expected_outcome="Braking is triggered.",
            )
        ],
        pytest_code="def test_threshold():\n    assert True\n",
    )


def _execution_result(
    *, exit_code: int, passed: int, failed: int, error_summary: str | None = None
) -> TestExecutionResult:
    return TestExecutionResult(
        exit_code=exit_code,
        passed=passed,
        failed=failed,
        duration_seconds=0.2,
        error_summary=error_summary,
        timed_out=False,
    )


class TestComputePassRate:
    def test_returns_none_when_denominator_is_zero(self) -> None:
        result = _execution_result(exit_code=1, passed=0, failed=0, error_summary="crashed")

        assert compute_pass_rate(result) is None

    def test_computes_ratio_when_denominator_positive(self) -> None:
        result = _execution_result(exit_code=1, passed=3, failed=1)

        assert compute_pass_rate(result) == pytest.approx(0.75)

    def test_returns_one_for_all_passing(self) -> None:
        result = _execution_result(exit_code=0, passed=4, failed=0)

        assert compute_pass_rate(result) == pytest.approx(1.0)


class TestBuildReport:
    def test_success_report_includes_first_run_pass_rate_only(self) -> None:
        state = AgentState(
            requirement=_sample_requirement(),
            knowledge_references=[_sample_reference()],
            test_plan=_sample_test_plan(),
            current_test_code=_sample_test_plan().pytest_code,
            execution_history=[_execution_result(exit_code=0, passed=1, failed=0)],
            correction_count=0,
            total_tokens=100,
            total_duration_seconds=1.0,
            final_status="success",
        )

        report = build_report(state)

        assert report.final_status == "success"
        assert report.final_test_code == state.current_test_code
        assert report.pass_rate_first_run == pytest.approx(1.0)
        assert report.pass_rate_after_correction is None
        assert report.summary
        assert report.error_message is None

    def test_report_after_correction_includes_both_pass_rates(self) -> None:
        state = AgentState(
            requirement=_sample_requirement(),
            knowledge_references=[_sample_reference()],
            test_plan=_sample_test_plan(),
            current_test_code=_sample_test_plan().pytest_code,
            execution_history=[
                _execution_result(exit_code=1, passed=0, failed=1, error_summary="boom"),
                _execution_result(exit_code=0, passed=1, failed=0),
            ],
            correction_count=1,
            total_tokens=200,
            total_duration_seconds=2.0,
            final_status="success",
        )

        report = build_report(state)

        assert report.pass_rate_first_run == pytest.approx(0.0)
        assert report.pass_rate_after_correction == pytest.approx(1.0)
        assert report.correction_count == 1

    def test_insufficient_info_report_allows_none_final_test_code(self) -> None:
        state = AgentState(
            requirement=_sample_requirement(),
            knowledge_references=[],
            correction_count=0,
            total_tokens=0,
            total_duration_seconds=0.0,
            final_status="insufficient_info",
        )

        report = build_report(state)

        assert report.final_status == "insufficient_info"
        assert report.final_test_code is None
        assert report.pass_rate_first_run is None
        assert report.pass_rate_after_correction is None
        assert report.summary

    def test_error_report_uses_provided_sanitized_error_message(self) -> None:
        state = AgentState(
            requirement=_sample_requirement(),
            knowledge_references=[_sample_reference()],
            correction_count=0,
            total_tokens=10,
            total_duration_seconds=0.1,
            final_status="error",
        )

        report = build_report(state, error_message="LLM 服务暂时不可用，请稍后重试。")

        assert report.final_status == "error"
        assert report.final_test_code is None
        assert report.error_message == "LLM 服务暂时不可用，请稍后重试。"
        assert "LLM 服务暂时不可用" in report.summary

    def test_raises_value_error_when_final_status_not_set(self) -> None:
        state = AgentState(requirement=_sample_requirement())

        with pytest.raises(ValueError, match="final_status"):
            build_report(state)
