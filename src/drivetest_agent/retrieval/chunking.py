"""Load and split Markdown knowledge documents into chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    section: str
    content: str


def load_knowledge_chunks(knowledge_dir: Path | str) -> list[KnowledgeChunk]:
    """Load all ``.md`` files under *knowledge_dir* and split into chunks."""
    root = Path(knowledge_dir)
    if not root.is_dir():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(_split_markdown(path.name, text))
    return chunks


def _split_markdown(source: str, text: str) -> list[KnowledgeChunk]:
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        return _paragraph_chunks(source, "文档", text)

    chunks: list[KnowledgeChunk] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        section_prefix = _section_prefix(matches, index, level, title)
        chunks.extend(_paragraph_chunks(source, section_prefix, body))
    return chunks


def _section_prefix(
    matches: list[re.Match[str]],
    index: int,
    level: int,
    title: str,
) -> str:
    parent_titles: list[str] = []
    for previous in reversed(matches[:index]):
        previous_level = len(previous.group(1))
        if previous_level < level:
            parent_titles.insert(0, previous.group(2).strip())
            level = previous_level
    parent_titles.append(title)
    return " > ".join(parent_titles)


def _paragraph_chunks(source: str, section: str, body: str) -> list[KnowledgeChunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if not paragraphs:
        return []
    return [
        KnowledgeChunk(source=source, section=section, content=paragraph)
        for paragraph in paragraphs
    ]
