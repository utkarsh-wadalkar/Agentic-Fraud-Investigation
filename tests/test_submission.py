from __future__ import annotations

from pathlib import Path


def test_submission_package_contains_required_operator_and_story_artifacts() -> None:
    required = {
        Path("docs/architecture.md"): "# Architecture",
        Path("docs/runbook.md"): "# Runbook",
        Path("submission/technical-post.md"): "# Technical Post",
        Path("submission/demo-script.md"): "# Demo Script",
        Path("submission/social-post.md"): "# Social Post",
    }

    for path, heading in required.items():
        assert path.is_file(), f"missing {path}"
        assert heading in path.read_text(encoding="utf-8")
