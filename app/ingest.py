"""Load knowledge-base markdown files, parse front matter, and chunk them.

Chunking strategy: split on level-2 (##) headings. Each chunk carries the
full document's front-matter metadata plus its own heading, so retrieval
and precedence logic can reason about status/authority per-chunk without
losing document-level context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    heading: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "unknown"))

    @property
    def policy_authority(self) -> str:
        return str(self.metadata.get("policy_authority", "unknown"))

    @property
    def is_authoritative(self) -> bool:
        """Only active, officially-authored docs may ground an answer."""
        return self.status == "active" and self.policy_authority == "official"

    def source_label(self) -> str:
        return f"{self.filename} \u2014 {self.heading}" if self.heading else self.filename


def _parse_front_matter(raw_text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text
    fm_text, body = match.groups()
    metadata = yaml.safe_load(fm_text) or {}
    return metadata, body


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, section_text) pairs on ## headings.

    Any text before the first ## heading (e.g. the H1 title line) is kept
    as its own section with an empty heading, so nothing is silently
    dropped from the index.
    """
    sections: list[tuple[str, str]] = []
    matches = list(HEADING_RE.finditer(body))

    if not matches:
        return [("", body.strip())]

    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        sections.append((heading, section_text))

    return sections


def load_documents(kb_dir: Path) -> list[Chunk]:
    """Load every .md file in kb_dir and return a flat list of chunks."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw)
        sections = _split_into_sections(body)
        for idx, (heading, section_text) in enumerate(sections):
            if not section_text:
                continue
            chunk_id = f"{path.name}::{idx}"
            # Prepend heading + doc title so the embedding captures context
            # even though only the section body follows it.
            title = metadata.get("title", path.stem)
            embed_text = f"{title}\n{heading}\n{section_text}" if heading else f"{title}\n{section_text}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    filename=path.name,
                    heading=heading,
                    text=embed_text,
                    metadata=metadata,
                )
            )
    return chunks
