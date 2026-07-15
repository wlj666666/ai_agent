"""Tests for Markdown loading and chunking."""

from __future__ import annotations

from pathlib import Path

from drivetest_agent.retrieval.chunking import KnowledgeChunk, load_knowledge_chunks


class TestMarkdownChunking:
    def test_loads_chunks_with_source_section_and_content(
        self, sample_markdown_dir: Path
    ) -> None:
        chunks = load_knowledge_chunks(sample_markdown_dir)

        assert len(chunks) >= 3
        assert all(isinstance(chunk, KnowledgeChunk) for chunk in chunks)
        assert all(chunk.source.endswith(".md") for chunk in chunks)
        assert all(chunk.section.strip() for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)

    def test_splits_by_headings_and_paragraphs(self, sample_markdown_dir: Path) -> None:
        chunks = load_knowledge_chunks(sample_markdown_dir)
        sections = {chunk.section for chunk in chunks}

        assert any("边界测试规范" in section for section in sections)
        assert any("TTC" in section for section in sections)
        assert any("异常输入" in section for section in sections)

        ttc_chunks = [chunk for chunk in chunks if "TTC" in chunk.section]
        assert any("1.5" in chunk.content for chunk in ttc_chunks)

    def test_empty_directory_returns_no_chunks(self, tmp_path: Path) -> None:
        assert load_knowledge_chunks(tmp_path) == []

    def test_ignores_non_markdown_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "spec.md").write_text("# 标题\n\n段落内容。\n", encoding="utf-8")

        chunks = load_knowledge_chunks(tmp_path)

        assert len(chunks) == 1
        assert chunks[0].source.endswith("spec.md")
