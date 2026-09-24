from __future__ import annotations

import importlib

from fraud_agent.graph import HashEmbedding


def module():
    return importlib.import_module("fraud_agent.knowledge")


def test_markdown_knowledge_chunks_are_stable_and_vector_ready(tmp_path) -> None:
    source = tmp_path / "README.md"
    source.write_text(
        "# Fraud handbook\n\n## Rule R5\n\nDetect small card-testing authorizations.\n\n"
        "## Evidence\n\nPrefer confirmed customer reports over device overlap.\n",
        encoding="utf-8",
    )

    chunks = module().read_markdown_chunks(source)
    vectors = module().vector_records(chunks, HashEmbedding(dimensions=8))

    assert len(chunks) == 3
    assert chunks[1].source_ref == "README.md#rule-r5"
    assert chunks[1].text == "## Rule R5\n\nDetect small card-testing authorizations."
    assert chunks == module().read_markdown_chunks(source)
    assert vectors[1]["vertex_id"] == chunks[1].chunk_id
    assert len(vectors[1]["vector"]) == 8
    assert vectors[1]["attributes"] == {
        "text": chunks[1].text,
        "source": "README.md",
        "source_ref": "README.md#rule-r5",
    }
