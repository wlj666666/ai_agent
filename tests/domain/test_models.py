"""Unit tests for core Pydantic domain models."""

import pytest
from pydantic import ValidationError

from drivetest_agent.domain.models import (
    AgentReport,
    AgentState,
    KnowledgeReference,
    Requirement,
    TestCasePlan,
    TestExecutionResult,
    TestPlan,
)


def _sample_requirement() -> Requirement:
    return Requirement(
        text="AEB triggers when TTC <= 1.5s and relative speed is positive.",
        component_description="Simulated AEB module",
    )


def _sample_reference() -> KnowledgeReference:
    return KnowledgeReference(
        source="docs/aeb-input-constraints.md",
        snippet="TTC must be non-negative and finite.",
        relevance_score=0.82,
    )


def _sample_test_case() -> TestCasePlan:
    return TestCasePlan(
        name="test_aeb_triggers_at_threshold",
        description="Verify braking at TTC exactly 1.5 seconds.",
        expected_outcome="Braking is triggered.",
    )


def _sample_test_plan() -> TestPlan:
    return TestPlan(
        test_cases=[_sample_test_case()],
        pytest_code="def test_threshold():\n    assert True\n",
    )


def _sample_execution_result() -> TestExecutionResult:
    return TestExecutionResult(
        exit_code=0,
        passed=3,
        failed=0,
        duration_seconds=0.42,
        error_summary=None,
        timed_out=False,
    )


class TestRequirementModel:
    def test_accepts_valid_requirement(self) -> None:
        req = _sample_requirement()
        assert req.text
        assert req.component_description == "Simulated AEB module"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            Requirement(text="")


class TestKnowledgeReferenceModel:
    def test_accepts_valid_reference(self) -> None:
        ref = _sample_reference()
        assert ref.source.endswith(".md")
        assert 0.0 <= ref.relevance_score <= 1.0

    def test_rejects_relevance_outside_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeReference(
                source="docs/spec.md",
                snippet="Boundary tests must cover both sides.",
                relevance_score=1.5,
            )

    def test_accepts_explicit_low_confidence_reference(self) -> None:
        reference = KnowledgeReference(
            source="docs/spec.md",
            snippet="No sufficiently relevant guidance was found.",
            relevance_score=0.05,
            low_confidence=True,
        )
        assert reference.low_confidence is True


class TestTestPlanModels:
    def test_test_case_plan_requires_name(self) -> None:
        with pytest.raises(ValidationError):
            TestCasePlan(
                name="",
                description="desc",
                expected_outcome="pass",
            )

    def test_test_plan_requires_at_least_one_case(self) -> None:
        with pytest.raises(ValidationError):
            TestPlan(test_cases=[], pytest_code="pass")

    def test_test_plan_rejects_empty_pytest_code(self) -> None:
        with pytest.raises(ValidationError):
            TestPlan(test_cases=[_sample_test_case()], pytest_code="")

    def test_test_plan_accepts_valid_payload(self) -> None:
        plan = _sample_test_plan()
        assert len(plan.test_cases) == 1
        assert "def test_threshold" in plan.pytest_code


class TestTestExecutionResultModel:
    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValidationError):
            TestExecutionResult(
                exit_code=1,
                passed=-1,
                failed=0,
                duration_seconds=0.1,
            )

    def test_accepts_timeout_flag(self) -> None:
        result = TestExecutionResult(
            exit_code=-1,
            passed=0,
            failed=1,
            duration_seconds=30.0,
            error_summary="Execution timed out.",
            timed_out=True,
        )
        assert result.timed_out is True


class TestAgentStateModel:
    @pytest.mark.parametrize("correction_count", [-1, 2])
    def test_rejects_correction_count_outside_allowed_range(
        self, correction_count: int
    ) -> None:
        with pytest.raises(ValidationError):
            AgentState(
                requirement=_sample_requirement(),
                correction_count=correction_count,
            )

    def test_rejects_negative_total_tokens(self) -> None:
        with pytest.raises(ValidationError):
            AgentState(requirement=_sample_requirement(), total_tokens=-1)

    def test_accepts_full_workflow_state(self) -> None:
        state = AgentState(
            requirement=_sample_requirement(),
            knowledge_references=[_sample_reference()],
            test_plan=_sample_test_plan(),
            current_test_code="def test_x(): pass",
            execution_history=[_sample_execution_result()],
            correction_count=1,
            total_tokens=1200,
            total_duration_seconds=3.5,
            final_status="success",
        )
        assert state.final_status == "success"
        assert state.total_tokens == 1200


class TestAgentReportModel:
    def test_requires_final_status(self) -> None:
        with pytest.raises(ValidationError):
            AgentReport(
                requirement=_sample_requirement(),
                knowledge_references=[],
                test_plan=None,
                execution_history=[],
                correction_count=0,
                total_tokens=0,
                total_duration_seconds=0.0,
                final_status="",
                final_test_code="def test_x(): pass",
                pass_rate_first_run=None,
                pass_rate_after_correction=None,
            )

    def test_rejects_empty_final_test_code(self) -> None:
        with pytest.raises(ValidationError):
            AgentReport(
                requirement=_sample_requirement(),
                correction_count=0,
                total_tokens=0,
                total_duration_seconds=0.0,
                final_status="success",
                final_test_code="",
            )

    def test_rejects_pass_rate_above_one(self) -> None:
        with pytest.raises(ValidationError):
            AgentReport(
                requirement=_sample_requirement(),
                correction_count=0,
                total_tokens=0,
                total_duration_seconds=0.0,
                final_status="success",
                final_test_code="def test_x(): pass",
                pass_rate_first_run=1.01,
            )

    @pytest.mark.parametrize("final_status", ["insufficient_info", "error"])
    def test_allows_none_final_test_code_when_no_code_was_generated(
        self, final_status: str
    ) -> None:
        report = AgentReport(
            requirement=_sample_requirement(),
            correction_count=0,
            total_tokens=0,
            total_duration_seconds=0.0,
            final_status=final_status,
            final_test_code=None,
        )
        assert report.final_test_code is None
        assert report.final_status == final_status

    @pytest.mark.parametrize("final_status", ["success", "failed"])
    def test_rejects_none_final_test_code_for_success_and_failed(
        self, final_status: str
    ) -> None:
        with pytest.raises(ValidationError):
            AgentReport(
                requirement=_sample_requirement(),
                correction_count=0,
                total_tokens=0,
                total_duration_seconds=0.0,
                final_status=final_status,
                final_test_code=None,
            )

    def test_accepts_report_with_pass_rates(self) -> None:
        report = AgentReport(
            requirement=_sample_requirement(),
            knowledge_references=[_sample_reference()],
            test_plan=_sample_test_plan(),
            execution_history=[_sample_execution_result()],
            correction_count=0,
            total_tokens=800,
            total_duration_seconds=2.1,
            final_status="success",
            final_test_code="def test_corrected_threshold():\n    assert True\n",
            pass_rate_first_run=1.0,
            pass_rate_after_correction=None,
        )
        assert report.pass_rate_first_run == 1.0
        assert "test_corrected_threshold" in report.final_test_code
