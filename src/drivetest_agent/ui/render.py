"""Streamlit rendering helpers.

These functions take already-computed view models / plain data and draw
Streamlit widgets. They contain no business wiring: no API clients, no
retriever, no agent construction. Importing this module never performs an
API call.
"""

from __future__ import annotations

import streamlit as st

from drivetest_agent.ui.examples import ExampleCase
from drivetest_agent.ui.view_model import ReportViewModel


def render_header() -> None:
    st.title("DriveTest Agent 演示")
    st.markdown(
        "输入智驾组件的需求/变更说明，系统会检索本地测试规范、生成 pytest 用例、"
        "受限执行并展示结果；首次执行失败时最多修正一次。"
    )


def render_example_picker(examples: list[ExampleCase]) -> ExampleCase | None:
    """Render the example selector and an apply button. Returns the chosen case."""
    if not examples:
        st.warning("未找到固定演示案例。")
        return None

    labels = [example.title for example in examples]
    selected_index = st.selectbox(
        "选择一个固定演示案例",
        options=range(len(examples)),
        format_func=lambda index: labels[index],
        key="selected_example_index",
    )
    selected = examples[selected_index]

    def _apply_example() -> None:
        st.session_state["requirement_text"] = selected.requirement
        st.session_state["component_description"] = selected.component_description or ""

    st.button("填充该案例到需求输入框", on_click=_apply_example, key="apply_example_button")
    return selected


def render_requirement_inputs() -> None:
    st.text_area(
        "需求 / 变更说明",
        key="requirement_text",
        height=140,
        placeholder="例如：AEB 模块新增：当 TTC 小于等于 1.5 秒且相对速度为正时触发制动。",
    )
    st.text_area(
        "组件说明（可选）",
        key="component_description",
        height=80,
        placeholder="例如：模拟 AEB 制动决策模块，输入为 TTC、相对速度和传感器有效状态。",
    )


def render_error(message: str) -> None:
    st.error(message)


def render_report(view: ReportViewModel) -> None:
    st.subheader(f"最终状态：{view.final_status_label}")
    st.write(f"摘要：{view.summary_display}")
    st.write(f"错误信息：{view.error_message_display}")

    _render_metrics(view)
    _render_references(view)
    _render_test_plan(view)
    _render_final_test_code(view)
    _render_executions(view)


def _render_metrics(view: ReportViewModel) -> None:
    columns = st.columns(5)
    columns[0].metric("首次通过率", view.pass_rate_first_run_display)
    columns[1].metric("修正后通过率", view.pass_rate_after_correction_display)
    columns[2].metric("修正次数", view.correction_count_display)
    columns[3].metric("总耗时", view.total_duration_display)
    columns[4].metric("Token 用量", view.total_tokens_display)


def _render_references(view: ReportViewModel) -> None:
    st.markdown("### 检索到的测试规范")
    if not view.references:
        st.write("无")
        return
    for reference in view.references:
        label = f"{reference.source}（相关度 {reference.score_display}）"
        if reference.low_confidence_label:
            label += f" · {reference.low_confidence_label}"
        with st.expander(label):
            st.write(reference.snippet)


def _render_test_plan(view: ReportViewModel) -> None:
    st.markdown("### 测试计划")
    if not view.has_test_plan or not view.test_plan_cases:
        st.write("无")
        return
    for case in view.test_plan_cases:
        st.markdown(f"**{case.name}**：{case.description}")
        st.caption(f"预期结果：{case.expected_outcome}")


def _render_final_test_code(view: ReportViewModel) -> None:
    st.markdown("### 最终 pytest 代码")
    if view.final_test_code_display == "无":
        st.write("无")
        return
    st.code(view.final_test_code_display, language="python")


def _render_executions(view: ReportViewModel) -> None:
    st.markdown("### 执行记录")
    if not view.executions:
        st.write("无")
        return
    for execution in view.executions:
        st.markdown(f"**{execution.label}**")
        st.write(
            f"passed={execution.passed} · failed={execution.failed} · "
            f"timed_out={execution.timed_out_label}"
        )
        st.text(f"错误摘要：{execution.error_summary_display}")
