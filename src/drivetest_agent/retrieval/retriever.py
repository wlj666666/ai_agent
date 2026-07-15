"""In-memory TF-IDF cosine retrieval over knowledge chunks."""

from __future__ import annotations

import math
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from drivetest_agent.domain.models import KnowledgeReference
from drivetest_agent.retrieval.chunking import KnowledgeChunk, load_knowledge_chunks


class KnowledgeRetriever:
    """Retrieve knowledge chunks using character n-gram TF-IDF similarity."""

    def __init__(
        self,
        knowledge_dir: Path | str,
        *,
        top_k: int = 3,
        low_confidence_threshold: float = 0.15,
        ngram_range: tuple[int, int] = (2, 4),
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")

        self._top_k = top_k
        self._low_confidence_threshold = low_confidence_threshold
        self._chunks: list[KnowledgeChunk] = load_knowledge_chunks(knowledge_dir)
        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
        corpus = [_chunk_text(chunk) for chunk in self._chunks]
        self._matrix = (
            self._vectorizer.fit_transform(corpus) if corpus else None
        )

    def search(self, query: str, *, top_k: int | None = None) -> list[KnowledgeReference]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        limit = self._top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1")
        if not self._chunks or self._matrix is None:
            return []

        query_vector = self._vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, self._matrix).flatten()
        for index, score in enumerate(scores):
            if not math.isfinite(float(score)):
                scores[index] = 0.0
        ranked_indices = scores.argsort()[::-1][:limit]

        max_score = max((float(scores[index]) for index in ranked_indices), default=0.0)
        low_confidence = max_score < self._low_confidence_threshold

        references: list[KnowledgeReference] = []
        for index in ranked_indices:
            score = float(scores[index])
            chunk = self._chunks[index]
            references.append(
                KnowledgeReference(
                    source=chunk.source,
                    snippet=_format_snippet(chunk),
                    relevance_score=min(max(score, 0.0), 1.0),
                    low_confidence=low_confidence,
                )
            )
        return references


def _chunk_text(chunk: KnowledgeChunk) -> str:
    return f"{chunk.section}\n{chunk.content}"


def _format_snippet(chunk: KnowledgeChunk) -> str:
    return f"[{chunk.section}] {chunk.content}"
