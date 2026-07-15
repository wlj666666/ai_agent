"""Prompt builders for test-plan generation and correction."""

from __future__ import annotations

from drivetest_agent.domain.models import KnowledgeReference, Requirement

_TEST_PLAN_SCHEMA_HINT = """\
Respond with a single JSON object matching this schema:
{
  "test_cases": [
    {
      "name": "test_<scenario>",
      "description": "...",
      "expected_outcome": "..."
    }
  ],
  "pytest_code": "<valid pytest module source code>"
}
Do not include markdown fences or commentary outside the JSON object."""


def _format_requirement(requirement: Requirement) -> str:
    lines = [requirement.text]
    if requirement.component_description:
        lines.append(f"Component: {requirement.component_description}")
    return "\n".join(lines)


def _format_knowledge(references: list[KnowledgeReference]) -> str:
    if not references:
        return "No knowledge references were provided."
    blocks: list[str] = []
    for reference in references:
        blocks.append(
            "\n".join(
                [
                    f"source: {reference.source}",
                    f"relevance_score: {reference.relevance_score:.2f}",
                    f"low_confidence: {reference.low_confidence}",
                    reference.snippet,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_generation_prompt(
    *,
    requirement: Requirement,
    knowledge_references: list[KnowledgeReference],
) -> str:
    """Build a prompt for initial structured test-plan generation."""
    return "\n\n".join(
        [
            "You are a test engineer for autonomous driving integration software.",
            "Generate a structured test plan and executable pytest code.",
            "=== REQUIREMENT ===",
            _format_requirement(requirement),
            "=== KNOWLEDGE ===",
            _format_knowledge(knowledge_references),
            _TEST_PLAN_SCHEMA_HINT,
        ]
    )


def build_correction_prompt(
    *,
    requirement: Requirement,
    knowledge_references: list[KnowledgeReference],
    previous_test_code: str,
    pytest_error: str,
) -> str:
    """Build a prompt for correcting pytest code after a failed run."""
    return "\n\n".join(
        [
            "You are a test engineer for autonomous driving integration software.",
            "Fix the pytest code based on the failure summary while preserving intent.",
            "=== REQUIREMENT ===",
            _format_requirement(requirement),
            "=== KNOWLEDGE ===",
            _format_knowledge(knowledge_references),
            "=== PREVIOUS_TEST ===",
            previous_test_code,
            "=== PYTEST_ERROR ===",
            pytest_error,
            _TEST_PLAN_SCHEMA_HINT,
        ]
    )
