"""Build user-facing AgentReport instances from finished AgentState."""

from __future__ import annotations

from drivetest_agent.domain.models import AgentReport, AgentState, TestExecutionResult

_SUCCESS_FIRST_RUN_SUMMARY = "生成的测试代码首次执行全部通过。"
_SUCCESS_AFTER_CORRECTION_SUMMARY = "首次执行未全部通过，修正后测试代码全部通过。"
_FAILED_SUMMARY = "首次执行失败，修正后仍未全部通过，已停止并生成失败报告。"
_INSUFFICIENT_INFO_SUMMARY = "检索到的规范信息不足或置信度过低，未生成测试代码。"
_DEFAULT_ERROR_SUMMARY = "模型服务调用失败，已终止本次生成。"


def compute_pass_rate(result: TestExecutionResult) -> float | None:
    """Return passed / (passed + failed), or None when the denominator is zero."""
    denominator = result.passed + result.failed
    if denominator == 0:
        return None
    return result.passed / denominator


def build_report(state: AgentState, *, error_message: str | None = None) -> AgentReport:
    """Construct the final AgentReport for a completed agent run.

    ``state`` must already have ``final_status`` set by the orchestrator.
    """
    if state.final_status is None:
        raise ValueError("state.final_status must be set before building a report")

    pass_rate_first_run = (
        compute_pass_rate(state.execution_history[0])
        if len(state.execution_history) >= 1
        else None
    )
    pass_rate_after_correction = (
        compute_pass_rate(state.execution_history[1])
        if len(state.execution_history) >= 2
        else None
    )

    return AgentReport(
        requirement=state.requirement,
        knowledge_references=state.knowledge_references,
        test_plan=state.test_plan,
        execution_history=state.execution_history,
        correction_count=state.correction_count,
        total_tokens=state.total_tokens,
        total_duration_seconds=state.total_duration_seconds,
        final_status=state.final_status,
        final_test_code=state.current_test_code,
        pass_rate_first_run=pass_rate_first_run,
        pass_rate_after_correction=pass_rate_after_correction,
        summary=_build_summary(state, error_message),
        error_message=error_message,
    )


def _build_summary(state: AgentState, error_message: str | None) -> str:
    if state.final_status == "success":
        if state.correction_count == 0:
            return _SUCCESS_FIRST_RUN_SUMMARY
        return _SUCCESS_AFTER_CORRECTION_SUMMARY
    if state.final_status == "failed":
        return _FAILED_SUMMARY
    if state.final_status == "insufficient_info":
        return _INSUFFICIENT_INFO_SUMMARY
    return error_message or _DEFAULT_ERROR_SUMMARY
