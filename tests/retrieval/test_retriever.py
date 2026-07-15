"""Tests for TF-IDF knowledge retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drivetest_agent.domain.models import KnowledgeReference
from drivetest_agent.retrieval.retriever import KnowledgeRetriever


class TestKnowledgeRetriever:
    def test_ttc_boundary_query_hits_boundary_spec(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("TTC 小于等于 1.5 秒 边界 阈值两侧")

        assert len(results) >= 1
        assert all(isinstance(item, KnowledgeReference) for item in results)
        assert results[0].source == "boundary-exception-testing.md"
        assert all(
            current.relevance_score >= following.relevance_score
            for current, following in zip(results, results[1:], strict=False)
        )

    def test_pytest_naming_query_hits_pytest_spec(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("pytest 测试函数命名 断言规范")

        assert len(results) >= 1
        assert results[0].source == "pytest-naming-assertions.md"

    def test_high_relevance_query_uses_default_confidence_threshold(
        self, knowledge_dir: Path
    ) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("TTC 1.5 秒 阈值两侧 1.49 1.51")

        assert len(results) >= 1
        assert results[0].low_confidence is False
        assert results[0].relevance_score >= 0.15

    def test_zero_overlap_query_uses_default_low_confidence_threshold(
        self, knowledge_dir: Path
    ) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("xyzqv")

        assert len(results) >= 1
        assert all(item.low_confidence for item in results)
        assert all(item.relevance_score == 0.0 for item in results)

    def test_non_finite_scores_become_zero_and_low_confidence(
        self, knowledge_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)

        def non_finite_scores(*_args: object, **_kwargs: object) -> np.ndarray:
            score_count = len(retriever._chunks)
            scores = np.zeros((1, score_count))
            scores[0, 0] = np.nan
            scores[0, 1] = np.inf
            return scores

        monkeypatch.setattr(
            "drivetest_agent.retrieval.retriever.cosine_similarity",
            non_finite_scores,
        )

        results = retriever.search("TTC")

        assert len(results) == 3
        assert all(item.relevance_score == 0.0 for item in results)
        assert all(item.low_confidence for item in results)

    def test_top_k_limits_result_count(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir, top_k=3)
        results = retriever.search("AEB 相对速度 传感器有效性", top_k=2)

        assert len(results) <= 2

    def test_default_top_k_is_three(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("AEB 输入约束 传感器")

        assert len(results) <= 3

    def test_empty_knowledge_directory_returns_empty_results(self, tmp_path: Path) -> None:
        retriever = KnowledgeRetriever(tmp_path)
        results = retriever.search("任意查询")

        assert results == []

    def test_empty_query_raises_value_error(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)

        with pytest.raises(ValueError, match="query"):
            retriever.search("")

    def test_whitespace_only_query_raises_value_error(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)

        with pytest.raises(ValueError, match="query"):
            retriever.search("   \t  ")

    def test_exception_input_query_hits_relevant_spec(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("异常输入 NaN 负值 无效传感器")

        assert len(results) >= 1
        joined = " ".join(item.snippet for item in results)
        assert "异常" in joined or "无效" in joined or "NaN" in joined

    def test_sensor_validity_query_hits_aeb_constraints(self, knowledge_dir: Path) -> None:
        retriever = KnowledgeRetriever(knowledge_dir)
        results = retriever.search("传感器有效性 sensor valid")

        assert len(results) >= 1
        joined = " ".join(item.snippet for item in results)
        assert "传感器" in joined or "有效" in joined
