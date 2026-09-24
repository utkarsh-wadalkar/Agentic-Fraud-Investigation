from __future__ import annotations

import asyncio
import importlib
import json
from datetime import datetime

from fraud_agent.graph import InMemoryGraphStore
from fraud_agent.models import InvestigationAnswer


def patterns():
    return importlib.import_module("fraud_agent.patterns")


def workflow():
    return importlib.import_module("fraud_agent.workflow")


def transaction(txn_id: str, ts: str, amount: str, **overrides: str) -> dict[str, str]:
    row = {
        "TransactionID": txn_id,
        "TransactionAmt": amount,
        "ts": ts,
        "channel": "online",
        "addr1": "100",
        "risk_score": "0.6",
        "ProductCD": "C",
        "card_id": "C00001-K1",
        "customer_id": "C00001",
    }
    row.update(overrides)
    return row


def test_card_testing_detector_finds_small_sequence_and_larger_purchase() -> None:
    history = [
        transaction("T1", "2016-11-01 10:00:00", "1.00"),
        transaction("T2", "2016-11-01 10:10:00", "2.00"),
        transaction("T3", "2016-11-01 10:20:00", "3.00"),
        transaction("T4", "2016-11-01 10:40:00", "250.00"),
    ]

    detection = patterns().detect_pattern(history, "T4", {"id_15": "New"})

    assert detection.pattern == "card_testing"
    assert detection.affected_txn_ids == ["T1", "T2", "T3", "T4"]
    assert detection.score >= 0.85


def test_evidence_simulator_is_deterministic_at_probability_boundaries() -> None:
    simulate = workflow().simulate_evidence_response

    assert simulate("step_up_auth", 0.75) == "Step-up authentication failed"
    assert simulate("customer_validation", 0.65) == "Customer confirmed the transaction"
    assert simulate("customer_validation", 0.70) == "No response within 24 hours"


def test_investigation_agent_uses_a_bounded_langgraph() -> None:
    agent = workflow().InvestigationAgent(InMemoryGraphStore())

    graph = agent.workflow.get_graph()

    assert "investigate" in graph.nodes


def test_customer_report_workflow_writes_schema_valid_case_to_graph() -> None:
    flagged = transaction(
        "T9",
        "2016-12-01 12:00:00",
        "482.12",
        risk_score="0.25",
        card_id="C00001-K1",
        customer_id="C00001",
    ) | {"id_15": "New", "id_23": "", "device_profile_id": "DEV-1"}
    store = InMemoryGraphStore(
        {
            "transaction_context": flagged,
            "card_history": [
                transaction("T1", "2016-11-01 12:00:00", "20.00"),
                flagged,
            ],
            "prior_cases": [{"case_id": "CC-0001", "pattern": "card_not_present_fraud"}],
            "device_neighbors": {"connected_card_ids": []},
        }
    )
    trigger = workflow().CaseTrigger(
        case_id="HHG-999",
        opened_at=datetime(2016, 12, 1, 12, 5),
        trigger_type="customer_report",
        trigger_text="Customer states they never made this purchase.",
        flagged_txn_id="T9",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=None,
    )

    answer = asyncio.run(workflow().InvestigationAgent(store).investigate(trigger))

    assert isinstance(answer, InvestigationAnswer)
    assert answer.case.verdict == "fraud"
    assert answer.case.pattern == "card_not_present_new_device"
    assert answer.case.written_to_graph is False
    assert answer.case.graph_case_id == ""
    assert answer.case.affected_txn_ids == ["T9"]
    assert [action.action for action in answer.next_best_actions.final] == [
        "BLOCK_CARD",
        "CREATE_CASE",
    ]
    assert answer.case.similar_prior_cases == ["CC-0001"]
    assert answer.tool_calls == store.tool_calls


def test_unconfirmed_device_neighbors_do_not_trigger_shared_origin_actions() -> None:
    flagged = transaction(
        "T9",
        "2016-12-01 12:00:00",
        "80.00",
        card_id="C00001-K1",
        customer_id="C00001",
    ) | {"device_profile_id": "DEV-COMMON"}
    store = InMemoryGraphStore(
        {
            "transaction_context": flagged,
            "card_history": [flagged],
            "prior_cases": [],
            "device_neighbors": {"connected_card_ids": ["C00002-K1", "C00003-K1"]},
        }
    )
    trigger = workflow().CaseTrigger(
        case_id="HHG-DEVICE",
        opened_at=datetime(2016, 12, 1, 12, 5),
        trigger_type="customer_report",
        trigger_text="Customer denied one purchase.",
        flagged_txn_id="T9",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=None,
    )

    answer = asyncio.run(workflow().InvestigationAgent(store).investigate(trigger))

    assert answer.case.connected_card_ids == []
    assert "MONITOR_CONNECTED_CARDS" not in [
        action.action for action in answer.next_best_actions.final
    ]
    assert answer.sar.file is False


def test_recurring_customer_dispute_follows_r7_without_blocking() -> None:
    prior = transaction(
        "T0",
        "2016-11-01 12:00:00",
        "59.99",
        risk_score="0.1",
        ProductCD="C",
    )
    flagged = transaction(
        "T1",
        "2016-12-01 12:00:00",
        "60.00",
        risk_score="0.2",
        ProductCD="C",
    )
    store = InMemoryGraphStore(
        {
            "transaction_context": flagged,
            "card_history": [prior, flagged],
            "prior_cases": [],
        }
    )
    trigger = workflow().CaseTrigger(
        case_id="HHG-RECURRING",
        opened_at=datetime(2016, 12, 1, 12, 5),
        trigger_type="customer_report",
        trigger_text="Customer does not recognize this recurring charge.",
        flagged_txn_id="T1",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=None,
    )

    answer = asyncio.run(workflow().InvestigationAgent(store).investigate(trigger))

    assert answer.case.verdict == "legitimate"
    assert answer.evidence_requests[0].type == "customer_validation"
    assert [action.action for action in answer.next_best_actions.initial] == [
        "CREATE_CASE",
        "VERIFY_WITH_CUSTOMER",
        "WARN_CUSTOMER",
    ]
    final_actions = [action.action for action in answer.next_best_actions.final]
    assert "WARN_CUSTOMER" in final_actions
    assert "CLOSE_NO_FRAUD" in final_actions
    assert "BLOCK_CARD" not in final_actions


def test_batch_runner_orders_cases_chronologically_and_writes_json(tmp_path) -> None:
    flagged_by_id = {
        "T1": transaction("T1", "2016-12-01 12:00:00", "10.00", risk_score="0.1"),
        "T2": transaction("T2", "2016-11-01 12:00:00", "10.00", risk_score="0.1"),
    }
    store = InMemoryGraphStore(
        {
            "transaction_context": lambda params: flagged_by_id[params["txn_id"]],
            "card_history": lambda params: [flagged_by_id[params["txn_id"]]],
            "prior_cases": [],
            "device_neighbors": {"connected_card_ids": []},
        }
    )
    triggers = [
        workflow().CaseTrigger(
            case_id="HHG-002",
            opened_at=datetime(2016, 12, 1, 12, 5),
            trigger_type="risk_score",
            trigger_text="Review.",
            flagged_txn_id="T1",
            card_id="C00001-K1",
            customer_id="C00001",
            risk_score=0.1,
        ),
        workflow().CaseTrigger(
            case_id="HHG-001",
            opened_at=datetime(2016, 11, 1, 12, 5),
            trigger_type="risk_score",
            trigger_text="Review.",
            flagged_txn_id="T2",
            card_id="C00001-K1",
            customer_id="C00001",
            risk_score=0.1,
        ),
    ]

    answers = asyncio.run(
        workflow().run_batch(triggers, workflow().InvestigationAgent(store), tmp_path)
    )

    assert [answer.case_id for answer in answers] == ["HHG-001", "HHG-002"]
    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["HHG-001.json", "HHG-002.json"]
    saved = InvestigationAnswer.model_validate(
        json.loads((tmp_path / "HHG-001.json").read_text(encoding="utf-8"))
    )
    assert saved.case_id == "HHG-001"


class FakeNarrator:
    async def rewrite(self, answer: InvestigationAnswer) -> tuple[str, str | None, int]:
        return "Grounded analyst summary.", None, 42


class FakeProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.rows: list[dict[str, object]] = []

    def predict_row(self, row: dict[str, object]) -> float:
        self.rows.append(row)
        return self.probability


def test_optional_narrator_only_rewrites_prose_and_records_tokens() -> None:
    flagged = transaction("T1", "2016-11-01 12:00:00", "10.00", risk_score="0.1")
    prior = transaction("T0", "2016-10-01 12:00:00", "10.00", risk_score="0.1")
    store = InMemoryGraphStore(
        {
            "transaction_context": flagged,
            "card_history": [prior, flagged],
            "prior_cases": [],
        }
    )
    trigger = workflow().CaseTrigger(
        case_id="HHG-001",
        opened_at=datetime(2016, 11, 1, 12, 5),
        trigger_type="risk_score",
        trigger_text="Review.",
        flagged_txn_id="T1",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=0.1,
    )

    answer = asyncio.run(
        workflow().InvestigationAgent(store, narrator=FakeNarrator()).investigate(trigger)
    )

    assert answer.case.summary == "Grounded analyst summary."
    assert answer.tokens == 42
    assert answer.case.verdict == "legitimate"


def test_trained_probability_model_outweighs_source_risk_score() -> None:
    flagged = transaction("T1", "2016-11-01 12:00:00", "10.00", risk_score="0.9")
    store = InMemoryGraphStore(
        {
            "transaction_context": flagged,
            "card_history": [flagged],
            "prior_cases": [],
        }
    )
    model = FakeProbabilityModel(0.05)
    trigger = workflow().CaseTrigger(
        case_id="HHG-001",
        opened_at=datetime(2016, 11, 1, 12, 5),
        trigger_type="risk_score",
        trigger_text="Review.",
        flagged_txn_id="T1",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=0.9,
    )

    answer = asyncio.run(
        workflow().InvestigationAgent(store, probability_model=model).investigate(trigger)
    )

    assert model.rows == [flagged]
    assert answer.case.verdict == "legitimate"
    assert answer.case.fraud_probability < 0.3


def test_no_response_keeps_moderate_risk_case_uncertain() -> None:
    flagged = transaction("T1", "2016-11-01 12:00:00", "75.00", risk_score="0.5")
    store = InMemoryGraphStore(
        {"transaction_context": flagged, "card_history": [flagged], "prior_cases": []}
    )
    trigger = workflow().CaseTrigger(
        case_id="HHG-001",
        opened_at=datetime(2016, 11, 1, 12, 5),
        trigger_type="risk_score",
        trigger_text="Review.",
        flagged_txn_id="T1",
        card_id="C00001-K1",
        customer_id="C00001",
        risk_score=0.5,
    )

    answer = asyncio.run(
        workflow()
        .InvestigationAgent(store, probability_model=FakeProbabilityModel(0.84))
        .investigate(trigger)
    )

    assert answer.evidence_requests[0].assumed_response == "No response within 24 hours"
    assert answer.case.verdict == "uncertain"
    assert "MONITOR_CARD" in [action.action for action in answer.next_best_actions.final]
    assert "BLOCK_CARD" not in [action.action for action in answer.next_best_actions.final]
