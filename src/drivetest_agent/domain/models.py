"""Pydantic models for agent workflow state and reporting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FinalStatus = Literal["success", "failed", "insufficient_info", "error"]


class Requirement(BaseModel):
    text: str = Field(min_length=1)
    component_description: str | None = None


class KnowledgeReference(BaseModel):
    source: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = False


class TestCasePlan(BaseModel):
    __test__ = False

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)


class TestPlan(BaseModel):
    __test__ = False

    test_cases: list[TestCasePlan] = Field(min_length=1)
    pytest_code: str = Field(min_length=1)


class TestExecutionResult(BaseModel):
    __test__ = False

    exit_code: int
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    error_summary: str | None = None
    timed_out: bool = False


class AgentState(BaseModel):
    requirement: Requirement
    knowledge_references: list[KnowledgeReference] = Field(default_factory=list)
    test_plan: TestPlan | None = None
    current_test_code: str | None = None
    execution_history: list[TestExecutionResult] = Field(default_factory=list)
    correction_count: int = Field(default=0, ge=0, le=1)
    total_tokens: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    final_status: FinalStatus | None = None


_STATUSES_REQUIRING_TEST_CODE = frozenset({"success", "failed"})


class AgentReport(BaseModel):
    requirement: Requirement
    knowledge_references: list[KnowledgeReference] = Field(default_factory=list)
    test_plan: TestPlan | None = None
    execution_history: list[TestExecutionResult] = Field(default_factory=list)
    correction_count: int = Field(ge=0, le=1)
    total_tokens: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0.0)
    final_status: FinalStatus
    final_test_code: str | None = None
    pass_rate_first_run: float | None = Field(default=None, ge=0.0, le=1.0)
    pass_rate_after_correction: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str | None = None
    error_message: str | None = None

    @field_validator("final_status")
    @classmethod
    def final_status_must_not_be_empty(cls, value: FinalStatus) -> FinalStatus:
        if not value:
            raise ValueError("final_status must not be empty")
        return value

    @model_validator(mode="after")
    def final_test_code_required_for_success_and_failed(self) -> AgentReport:
        if self.final_status in _STATUSES_REQUIRING_TEST_CODE and not self.final_test_code:
            raise ValueError(
                "final_test_code must be a non-empty string when final_status is "
                "'success' or 'failed'"
            )
        return self
