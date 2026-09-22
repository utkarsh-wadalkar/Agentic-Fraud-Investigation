from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from fraud_agent.models import (
    CaseRecord,
    EvidenceItem,
    InvestigationAnswer,
    NextBestActions,
    RecommendedAction,
    SarRecord,
)
from fraud_agent.policy import approval_route, validate_policy_consistency


def make_answer() -> InvestigationAnswer:
    return InvestigationAnswer(
        case_id="HHG-001",
        case=CaseRecord(
            status="closed_fraud",
            verdict="fraud",
            fraud_probability=0.91,
            pattern="card_testing",
            affected_txn_ids=["3514030", "3514031"],
            first_suspicious_txn_id="3514030",
            connected_card_ids=[],
            connected_device_profiles=[],
            exposure_usd=127.07,
            evidence=[
                EvidenceItem(
                    claim="Testing sequence observed",
                    source="graph",
                    ref="query:card_testing",
                    entity_ids=["3514030", "3514031"],
                )
            ],
            similar_prior_cases=["CC-0001"],
            summary="A testing sequence preceded a larger transaction.",
            written_to_graph=True,
            graph_case_id="HHG-001",
        ),
        evidence_requests=[],
        next_best_actions=NextBestActions(
            initial=[
                RecommendedAction(
                    action="BLOCK_CARD", route="L1", reason="R5: exposure is below $2,500"
                ),
                RecommendedAction(action="CREATE_CASE", route="auto", reason="R3a"),
            ],
            final=[
                RecommendedAction(
                    action="BLOCK_CARD", route="L1", reason="R5: exposure is below $2,500"
                ),
                RecommendedAction(action="CREATE_CASE", route="auto", reason="R3a"),
            ],
            what_changed="nothing",
        ),
        sar=SarRecord(
            file=False,
            reason="Exposure does not meet reporting conditions.",
            narrative="",
            subjects=[],
            total_amount_usd=0,
            activity_dates=[],
        ),
        stop_reason="Threshold met with independent evidence.",
        tool_calls=7,
        tokens=1200,
        latency_s=2.5,
    )


def test_legitimate_answer_rejects_affected_transactions() -> None:
    payload = make_answer().model_dump()
    payload["case"]["verdict"] = "legitimate"
    payload["case"]["status"] = "closed_legitimate"

    with pytest.raises(ValidationError, match="legitimate case"):
        InvestigationAnswer.model_validate(payload)


def test_answer_rejects_sar_action_mismatch() -> None:
    payload = make_answer().model_dump()
    payload["sar"] = {
        "file": True,
        "reason": "R6",
        "narrative": "Six sentences are generated later.",
        "subjects": ["C12382"],
        "total_amount_usd": 127.07,
        "activity_dates": ["2016-12-04", "2016-12-04"],
    }

    with pytest.raises(ValidationError, match="FILE_REPORT"):
        InvestigationAnswer.model_validate(payload)


def test_answer_requires_initial_and_final_to_match_without_requests() -> None:
    payload = make_answer().model_dump()
    payload["next_best_actions"]["final"].append(
        {"action": "MONITOR_CARD", "route": "auto", "reason": "R4"}
    )

    with pytest.raises(ValidationError, match="must match"):
        InvestigationAnswer.model_validate(payload)


@pytest.mark.parametrize(
    ("action", "exposure", "expected"),
    [
        ("ALLOW_TRANSACTION", 10, "auto"),
        ("DECLINE_TRANSACTION", 10, "L1"),
        ("BLOCK_CARD", 2500, "L1"),
        ("BLOCK_CARD", 2500.01, "L2"),
        ("BLOCK_ALL_CARDS", 10, "L2"),
        ("FILE_REPORT", 10, "L2"),
    ],
)
def test_approval_route_matches_policy(action: str, exposure: float, expected: str) -> None:
    assert approval_route(action, exposure) == expected


def test_policy_consistency_rejects_wrong_route() -> None:
    answer = make_answer()
    answer.next_best_actions.final[0].route = "auto"

    errors = validate_policy_consistency(answer)

    assert "BLOCK_CARD must use L1 at exposure 127.07" in errors


def test_settings_reads_existing_anthropic_and_tigergraph_environment(monkeypatch) -> None:
    monkeypatch.setenv("TG_HOST", "https://example.i.tgcloud.io")
    monkeypatch.setenv("TG_GRAPHNAME", "FraudGraph")
    monkeypatch.setenv("TG_API_TOKEN", "secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    monkeypatch.setenv("ANTHROPIC_MODEL", "combo")
    settings_type = importlib.import_module("fraud_agent.config").Settings

    settings = settings_type()

    assert settings.tigergraph_ready is True
    assert settings.anthropic_ready is True
    assert settings.source_dir.name == "Drive Files"
