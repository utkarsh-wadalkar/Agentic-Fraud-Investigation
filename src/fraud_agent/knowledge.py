"""Prepare deterministic Markdown chunks for TigerGraph vector retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    source: str
    source_ref: str


def _slug(heading: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return value or "document"


def _chunk_id(source_ref: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_ref}\n{text}".encode()).hexdigest()[:16].upper()
    return f"K-{digest}"


def read_markdown_chunks(path: Path, max_chars: int = 1_800) -> list[KnowledgeChunk]:
    """Split Markdown by heading and paragraph while retaining stable source references."""
    sections: list[tuple[str, list[str]]] = []
    heading = "document"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            if any(item.strip() for item in lines):
                sections.append((heading, lines))
            heading = line.lstrip("#").strip()
            lines = [line]
        else:
            lines.append(line)
    if any(item.strip() for item in lines):
        sections.append((heading, lines))

    chunks: list[KnowledgeChunk] = []
    for section_heading, section_lines in sections:
        paragraphs = [
            part.strip()
            for part in "\n".join(section_lines).split("\n\n")
            if part.strip()
        ]
        batch: list[str] = []
        batch_length = 0
        part_number = 1
        for paragraph in paragraphs:
            added = len(paragraph) + (2 if batch else 0)
            if batch and batch_length + added > max_chars:
                text = "\n\n".join(batch)
                ref = f"{path.name}#{_slug(section_heading)}"
                if part_number > 1:
                    ref += f"-{part_number}"
                chunks.append(KnowledgeChunk(_chunk_id(ref, text), text, path.name, ref))
                batch = []
                batch_length = 0
                part_number += 1
            batch.append(paragraph)
            batch_length += len(paragraph) + (2 if len(batch) > 1 else 0)
        if batch:
            text = "\n\n".join(batch)
            ref = f"{path.name}#{_slug(section_heading)}"
            if part_number > 1:
                ref += f"-{part_number}"
            chunks.append(KnowledgeChunk(_chunk_id(ref, text), text, path.name, ref))
    return chunks


def vector_records(chunks: list[KnowledgeChunk], embedder: Embedder) -> list[dict[str, object]]:
    """Convert knowledge chunks to the MCP vector-upsert payload shape."""
    return [
        {
            "vertex_id": chunk.chunk_id,
            "vector": embedder.embed(chunk.text),
            "attributes": {
                "text": chunk.text,
                "source": chunk.source,
                "source_ref": chunk.source_ref,
            },
        }
        for chunk in chunks
    ]
