"""Pure conversion from ``AgentReport`` into rendering-friendly display data.

No Streamlit import here: every function is plain Python so it can be unit
tested without a running app, a network call, or an API key. All ``None``
values are converted to a safe placeholder string instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drivetest_agent.domain.models import AgentReport, FinalStatus, TestExecutionResult

NONE_PLACEHOLDER = "无"
NOT_EXECUTED_PLACEHOLDER = "未执行"

_STATUS_LABELS: dict[FinalStatus, str] = {
    "success": "成功",
    "failed": "失败",
    "insufficient_info": "信息不足",
    "error": "错误",
}

_EXECUTION_LABELS = ("首次执行", "修正后执行")


@dataclass(frozen=True)
class ReferenceView:
    source: str
    snippet: str
    score_display: str
    low_confidence: bool
    low_confidence_label: str


@dataclass(frozen=True)
class TestCaseView:
    name: str
    description: str
    expected_outcome: str


@dataclass(frozen=True)
class ExecutionView:
    label: str
    passed: int
    failed: int
    timed_out: bool
    timed_out_label: str
    error_summary_display: str


@dataclass(frozen=True)
class ReportViewModel:
    final_status: FinalStatus
    final_status_label: str
    summary_display: str
    error_message_display: str
    references: list[ReferenceView] = field(default_factory=list)
    has_test_plan: bool = False
    test_plan_cases: list[TestCaseView] = field(default_factory=list)
    final_test_code_display: str = NONE_PLACEHOLDER
    executions: list[ExecutionView] = field(default_factory=list)
    pass_rate_first_run_display: str = NOT_EXECUTED_PLACEHOLDER
    pass_rate_after_correction_display: str = NOT_EXECUTED_PLACEHOLDER
    total_duration_display: str = "0.00 秒"
    total_tokens_display: str = "0"
    correction_count_display: str = "0 次"


def build_report_view_model(report: AgentReport) -> ReportViewModel:
    """Convert a completed ``AgentReport`` into a safe-to-render view model."""
    return ReportViewModel(
        final_status=report.final_status,
        final_status_label=_STATUS_LABELS.get(report.final_status, str(report.final_status)),
        summary_display=_or_placeholder(report.summary, NONE_PLACEHOLDER),
        error_message_display=_or_placeholder(report.error_message, NONE_PLACEHOLDER),
        references=[_build_reference_view(reference) for reference in report.knowledge_references],
        has_test_plan=report.test_plan is not None,
        test_plan_cases=_build_test_case_views(report),
        final_test_code_display=_or_placeholder(report.final_test_code, NONE_PLACEHOLDER),
        executions=[
            _build_execution_view(index, execution)
            for index, execution in enumerate(report.execution_history)
        ],
        pass_rate_first_run_display=_format_pass_rate(report.pass_rate_first_run),
        pass_rate_after_correction_display=_format_pass_rate(report.pass_rate_after_correction),
        total_duration_display=f"{report.total_duration_seconds:.2f} 秒",
        total_tokens_display=str(report.total_tokens),
        correction_count_display=f"{report.correction_count} 次",
    )


def _or_placeholder(value: str | None, placeholder: str) -> str:
    if value is None or not value.strip():
        return placeholder
    return value


def _format_pass_rate(rate: float | None) -> str:
    if rate is None:
        return NOT_EXECUTED_PLACEHOLDER
    return f"{round(rate * 100)}%"


def _build_reference_view(reference: object) -> ReferenceView:
    source = getattr(reference, "source", "")
    snippet = getattr(reference, "snippet", "")
    relevance_score = getattr(reference, "relevance_score", 0.0)
    low_confidence = bool(getattr(reference, "low_confidence", False))
    return ReferenceView(
        source=source,
        snippet=snippet,
        score_display=f"{relevance_score:.2f}",
        low_confidence=low_confidence,
        low_confidence_label="低置信度" if low_confidence else "",
    )


def _build_test_case_views(report: AgentReport) -> list[TestCaseView]:
    if report.test_plan is None:
        return []
    return [
        TestCaseView(
            name=test_case.name,
            description=test_case.description,
            expected_outcome=test_case.expected_outcome,
        )
        for test_case in report.test_plan.test_cases
    ]


def _build_execution_view(index: int, execution: TestExecutionResult) -> ExecutionView:
    label = _EXECUTION_LABELS[index] if index < len(_EXECUTION_LABELS) else f"第 {index + 1} 次执行"
    return ExecutionView(
        label=label,
        passed=execution.passed,
        failed=execution.failed,
        timed_out=execution.timed_out,
        timed_out_label="是" if execution.timed_out else "否",
        error_summary_display=_or_placeholder(execution.error_summary, NONE_PLACEHOLDER),
    )
