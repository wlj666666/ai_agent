"""Tests for LLM prompt builders."""

from __future__ import annotations

from drivetest_agent.domain.models import KnowledgeReference, Requirement, TestPlan
from drivetest_agent.llm.prompts import build_correction_prompt, build_generation_prompt


def _sample_requirement() -> Requirement:
    return Requirement(
        text="AEB triggers when TTC <= 1.5s and relative speed is positive.",
        component_description="Simulated AEB module",
    )


def _sample_references() -> list[KnowledgeReference]:
    return [
        KnowledgeReference(
            source="knowledge/aeb-input-constraints.md",
            snippet="TTC must be non-negative and finite.",
            relevance_score=0.82,
        ),
        KnowledgeReference(
            source="knowledge/boundary-exception-testing.md",
            snippet="Cover both sides of numeric thresholds.",
            relevance_score=0.71,
        ),
    ]


class TestGenerationPrompt:
    def test_includes_requirement_knowledge_and_schema_sections(self) -> None:
        prompt = build_generation_prompt(
            requirement=_sample_requirement(),
            knowledge_references=_sample_references(),
        )

        assert "=== REQUIREMENT ===" in prompt
        assert "=== KNOWLEDGE ===" in prompt
        assert "TTC <= 1.5s" in prompt
        assert "Simulated AEB module" in prompt

    def test_knowledge_includes_source_per_reference(self) -> None:
        prompt = build_generation_prompt(
            requirement=_sample_requirement(),
            knowledge_references=_sample_references(),
        )

        assert "source: knowledge/aeb-input-constraints.md" in prompt
        assert "source: knowledge/boundary-exception-testing.md" in prompt
        assert "TTC must be non-negative and finite." in prompt

    def test_constrains_test_plan_json_output(self) -> None:
        prompt = build_generation_prompt(
            requirement=_sample_requirement(),
            knowledge_references=_sample_references(),
        )

        assert "test_cases" in prompt
        assert "pytest_code" in prompt
        assert "JSON" in prompt.upper()


class TestCorrectionPrompt:
    def test_includes_all_required_sections(self, sample_test_plan: TestPlan) -> None:
        plan = sample_test_plan
        truncated_error = "AssertionError: expected True\n... [truncated]"

        prompt = build_correction_prompt(
            requirement=_sample_requirement(),
            knowledge_references=_sample_references(),
            previous_test_code=plan.pytest_code,
            pytest_error=truncated_error,
        )

        assert "=== REQUIREMENT ===" in prompt
        assert "=== KNOWLEDGE ===" in prompt
        assert "=== PREVIOUS_TEST ===" in prompt
        assert "=== PYTEST_ERROR ===" in prompt
        assert plan.pytest_code in prompt
        assert truncated_error in prompt

    def test_does_not_repeat_full_knowledge_documents(self) -> None:
        prompt = build_correction_prompt(
            requirement=_sample_requirement(),
            knowledge_references=_sample_references(),
            previous_test_code="def test_x(): pass",
            pytest_error="failed",
        )

        assert prompt.count("=== KNOWLEDGE ===") == 1
        assert "source: knowledge/aeb-input-constraints.md" in prompt
