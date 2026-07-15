"""Pure conversion from AgentReport to a rendering-friendly view model.

These tests must not require Streamlit, a network call, or any API key.
"""

from __future__ import annotations

from drivetest_agent.domain.models import (
    AgentReport,
    KnowledgeReference,
    Requirement,
    TestCasePlan,
    TestExecutionResult,
    TestPlan,
)
from drivetest_agent.ui.view_model import (
    NONE_PLACEHOLDER,
    NOT_EXECUTED_PLACEHOLDER,
    build_report_view_model,
)

_REQUIREMENT = Requirement(text="AEB 模块新增：当 TTC 小于等于 1.5 秒且相对速度为正时触发制动。")


def _passing_execution(**overrides: object) -> TestExecutionResult:
    defaults: dict[str, object] = dict(
        exit_code=0, passed=1, failed=0, duration_seconds=0.1, error_summary=None, timed_out=False
    )
    defaults.update(overrides)
    return TestExecutionResult(**defaults)


def _failing_execution(**overrides: object) -> TestExecutionResult:
    defaults: dict[str, object] = dict(
        exit_code=1,
        passed=0,
        failed=1,
        duration_seconds=0.2,
        error_summary="AssertionError: boom",
        timed_out=False,
    )
    defaults.update(overrides)
    return TestExecutionResult(**defaults)


def _test_plan() -> TestPlan:
    return TestPlan(
        test_cases=[
            TestCasePlan(
                name="test_boundary",
                description="覆盖 TTC 等号边界。",
                expected_outcome="TTC=1.5 时触发制动。",
            )
        ],
        pytest_code="def test_boundary():\n    assert True\n",
    )


class TestSuccessFirstRun:
    def test_builds_view_model_without_correction(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[
                KnowledgeReference(
                    source="knowledge/aeb-input-constraints.md",
                    snippet="TTC 必须为非负有限数值。",
                    relevance_score=0.42,
                    low_confidence=False,
                )
            ],
            test_plan=_test_plan(),
            execution_history=[_passing_execution()],
            correction_count=0,
            total_tokens=120,
            total_duration_seconds=0.55,
            final_status="success",
            final_test_code="def test_boundary():\n    assert True\n",
            pass_rate_first_run=1.0,
            pass_rate_after_correction=None,
            summary="生成的测试代码首次执行全部通过。",
            error_message=None,
        )

        view = build_report_view_model(report)

        assert view.final_status == "success"
        assert "成功" in view.final_status_label
        assert view.summary_display == "生成的测试代码首次执行全部通过。"
        assert view.error_message_display == NONE_PLACEHOLDER
        assert view.has_test_plan is True
        assert len(view.test_plan_cases) == 1
        assert view.test_plan_cases[0].name == "test_boundary"
        assert view.final_test_code_display == "def test_boundary():\n    assert True\n"
        assert len(view.executions) == 1
        assert view.executions[0].label == "首次执行"
        assert view.executions[0].error_summary_display == NONE_PLACEHOLDER
        assert view.executions[0].timed_out_label == "否"
        assert view.pass_rate_first_run_display == "100%"
        assert view.pass_rate_after_correction_display == NOT_EXECUTED_PLACEHOLDER
        assert view.correction_count_display == "0 次"
        assert "0.55" in view.total_duration_display
        assert view.total_tokens_display == "120"

        reference_view = view.references[0]
        assert reference_view.source == "knowledge/aeb-input-constraints.md"
        assert reference_view.score_display == "0.42"
        assert reference_view.low_confidence_label == ""


class TestSuccessAfterCorrection:
    def test_builds_two_execution_entries_with_labels(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[],
            test_plan=_test_plan(),
            execution_history=[_failing_execution(), _passing_execution()],
            correction_count=1,
            total_tokens=200,
            total_duration_seconds=0.75,
            final_status="success",
            final_test_code="def test_boundary():\n    assert True\n",
            pass_rate_first_run=0.0,
            pass_rate_after_correction=1.0,
            summary="首次执行未全部通过，修正后测试代码全部通过。",
            error_message=None,
        )

        view = build_report_view_model(report)

        assert [execution.label for execution in view.executions] == ["首次执行", "修正后执行"]
        assert view.executions[0].error_summary_display == "AssertionError: boom"
        assert view.pass_rate_first_run_display == "0%"
        assert view.pass_rate_after_correction_display == "100%"
        assert view.correction_count_display == "1 次"


class TestFailedAfterCorrection:
    def test_marks_final_status_as_failed(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[],
            test_plan=_test_plan(),
            execution_history=[_failing_execution(), _failing_execution()],
            correction_count=1,
            total_tokens=200,
            total_duration_seconds=0.75,
            final_status="failed",
            final_test_code="def test_boundary():\n    assert True\n",
            pass_rate_first_run=0.0,
            pass_rate_after_correction=0.0,
            summary="首次执行失败，修正后仍未全部通过，已停止并生成失败报告。",
            error_message=None,
        )

        view = build_report_view_model(report)

        assert view.final_status == "failed"
        assert "失败" in view.final_status_label
        assert view.pass_rate_first_run_display == "0%"
        assert view.pass_rate_after_correction_display == "0%"


class TestInsufficientInfo:
    def test_handles_missing_test_plan_and_executions_and_pass_rates(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[
                KnowledgeReference(
                    source="knowledge/aeb-input-constraints.md",
                    snippet="弱匹配片段。",
                    relevance_score=0.02,
                    low_confidence=True,
                )
            ],
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

        view = build_report_view_model(report)

        assert view.final_status == "insufficient_info"
        assert view.has_test_plan is False
        assert view.test_plan_cases == []
        assert view.final_test_code_display == NONE_PLACEHOLDER
        assert view.executions == []
        assert view.pass_rate_first_run_display == NOT_EXECUTED_PLACEHOLDER
        assert view.pass_rate_after_correction_display == NOT_EXECUTED_PLACEHOLDER
        assert view.references[0].low_confidence_label != ""


class TestErrorStatusDoesNotCrashOnNoneFields:
    def test_handles_none_test_plan_none_error_message_and_no_executions(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[],
            test_plan=None,
            execution_history=[],
            correction_count=0,
            total_tokens=0,
            total_duration_seconds=0.0,
            final_status="error",
            final_test_code=None,
            pass_rate_first_run=None,
            pass_rate_after_correction=None,
            summary=None,
            error_message="调用模型服务失败，请稍后重试。",
        )

        view = build_report_view_model(report)

        assert view.final_status == "error"
        assert view.summary_display == NONE_PLACEHOLDER
        assert view.error_message_display == "调用模型服务失败，请稍后重试。"
        assert view.has_test_plan is False
        assert view.final_test_code_display == NONE_PLACEHOLDER
        assert view.executions == []

    def test_handles_error_status_with_none_error_message_without_raising(self) -> None:
        report = AgentReport(
            requirement=_REQUIREMENT,
            knowledge_references=[],
            test_plan=None,
            execution_history=[],
            correction_count=0,
            total_tokens=0,
            total_duration_seconds=0.0,
            final_status="error",
            final_test_code=None,
            pass_rate_first_run=None,
            pass_rate_after_correction=None,
            summary=None,
            error_message=None,
        )

        view = build_report_view_model(report)

        assert view.error_message_display == NONE_PLACEHOLDER
