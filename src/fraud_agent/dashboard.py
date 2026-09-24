"""Pure dashboard view models shared by Streamlit and smoke tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fraud_agent.models import InvestigationAnswer

GraphEdge = tuple[str, str, str]


@dataclass(frozen=True)
class DashboardView:
    answers: dict[str, InvestigationAnswer]
    total_cases: int
    verdict_counts: dict[str, int]
    total_exposure: float
    sar_count: int
    average_latency: float
    case_graph: dict[str, list[GraphEdge]]


def build_dashboard_view(cases_dir: Path) -> DashboardView:
    answers: dict[str, InvestigationAnswer] = {}
    for path in sorted(cases_dir.glob("HHG-*.json")):
        answer = InvestigationAnswer.model_validate(json.loads(path.read_text(encoding="utf-8")))
        answers[answer.case_id] = answer

    verdict_counts = {"fraud": 0, "legitimate": 0, "uncertain": 0}
    graph: dict[str, list[GraphEdge]] = {}
    for case_id, answer in answers.items():
        verdict_counts[answer.case.verdict] += 1
        edges = [(case_id, card_id, "connected card") for card_id in answer.case.connected_card_ids]
        edges.extend(
            (case_id, device_id, "device") for device_id in answer.case.connected_device_profiles
        )
        graph[case_id] = edges

    count = len(answers)
    return DashboardView(
        answers=answers,
        total_cases=count,
        verdict_counts=verdict_counts,
        total_exposure=round(sum(item.case.exposure_usd for item in answers.values()), 2),
        sar_count=sum(item.sar.file for item in answers.values()),
        average_latency=(
            round(sum(item.latency_s for item in answers.values()) / count, 3) if count else 0.0
        ),
        case_graph=graph,
    )
