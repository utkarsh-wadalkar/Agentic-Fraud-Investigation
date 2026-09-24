"""Guarded, evidence-first investigation workflow."""

# ruff: noqa: E501 -- policy and regulatory prose remain auditable as literals.

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from fraud_agent.graph import HashEmbedding
from fraud_agent.models import (
    CaseRecord,
    EvidenceItem,
    EvidenceRequest,
    InvestigationAnswer,
    NextBestActions,
    RecommendedAction,
    SarRecord,
)
from fraud_agent.patterns import PatternDetection, detect_pattern, recurring_charge_matches
from fraud_agent.policy import approval_route, validate_policy_consistency

TriggerType = Literal["risk_score", "customer_report", "analyst_request"]


@dataclass(frozen=True)
class CaseTrigger:
    case_id: str
    opened_at: datetime
    trigger_type: TriggerType
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: float | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> CaseTrigger:
        return cls(
            case_id=row["case_id"],
            opened_at=datetime.fromisoformat(row["opened_at"]),
            trigger_type=row["trigger_type"],  # type: ignore[arg-type]
            trigger_text=row["trigger_text"],
            flagged_txn_id=row["flagged_txn_id"],
            card_id=row["card_id"],
            customer_id=row["customer_id"],
            risk_score=float(row["risk_score"]) if row.get("risk_score") else None,
        )


class GraphStore(Protocol):
    tool_calls: int
    writes_to_tigergraph: bool

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any: ...

    async def search_knowledge(self, vector: list[float], top_k: int = 5) -> Any: ...

    async def write_case(
        self, answer: InvestigationAnswer, card_id: str, opened_at: datetime
    ) -> str: ...


class Narrator(Protocol):
    async def rewrite(self, answer: InvestigationAnswer) -> tuple[str, str | None, int]: ...


class ProbabilityModel(Protocol):
    def predict_row(self, row: dict[str, Any]) -> float: ...


class InvestigationState(TypedDict):
    trigger: CaseTrigger
    answer: InvestigationAnswer | None


def simulate_evidence_response(request_type: str, probability: float) -> str:
    if request_type == "analyst_info":
        return "No additional evidence available"
    if probability >= 0.75:
        return (
            "Step-up authentication failed"
            if request_type == "step_up_auth"
            else "Customer denied the transaction"
        )
    if probability <= 0.65:
        return (
            "Step-up authentication passed"
            if request_type == "step_up_auth"
            else "Customer confirmed the transaction"
        )
    return "No response within 24 hours"


def _probability(
    trigger: CaseTrigger,
    flagged: dict[str, Any],
    detection: PatternDetection,
    model: ProbabilityModel | None = None,
) -> float:
    source_score = trigger.risk_score
    if source_score is None:
        source_score = float(flagged.get("risk_score") or 0.5)
    if model is None:
        probability = max(source_score, detection.score)
    else:
        probability = (
            0.65 * model.predict_row(flagged) + 0.20 * detection.score + 0.15 * source_score
        )
    if trigger.trigger_type == "customer_report":
        probability = max(probability, 0.88)
    if trigger.trigger_type == "analyst_request":
        probability = max(probability, 0.60)
    if flagged.get("id_15") == "New":
        probability += 0.04
    if flagged.get("id_23"):
        probability += 0.05
    return min(0.99, max(0.01, round(probability, 4)))


def _updated_probability(probability: float, response: str) -> float:
    if "denied" in response.lower() or "failed" in response.lower():
        return min(0.99, probability + 0.22)
    if "confirmed" in response.lower() or "passed" in response.lower():
        return max(0.01, probability - 0.45)
    return probability


def _recommendation(action: str, exposure: float, reason: str) -> RecommendedAction:
    return RecommendedAction(
        action=action,  # type: ignore[arg-type]
        route=approval_route(action, exposure),
        reason=reason,
    )


def _fraud_actions(exposure: float, connected: list[str]) -> list[RecommendedAction]:
    actions = [
        _recommendation(
            "BLOCK_CARD", exposure, "R2: unauthorized activity is confirmed or strongly suspected"
        ),
        _recommendation("CREATE_CASE", exposure, "R2 and R3a: preserve the investigation record"),
    ]
    if connected:
        actions.append(
            _recommendation(
                "MONITOR_CONNECTED_CARDS", exposure, "R6: cards share an investigation origin"
            )
        )
    if exposure > 1000 or connected:
        actions.append(
            _recommendation("FILE_REPORT", exposure, "R2/R6: reporting threshold is met")
        )
    return actions


def _actions(
    trigger: CaseTrigger,
    probability: float,
    exposure: float,
    connected: list[str],
    request: EvidenceRequest | None,
    response: str | None = None,
    recurring_dispute: bool = False,
) -> list[RecommendedAction]:
    if recurring_dispute:
        if response and ("confirmed" in response.lower() or "passed" in response.lower()):
            return [
                _recommendation("ALLOW_TRANSACTION", exposure, "R7: recurring charge confirmed"),
                _recommendation("CREATE_CASE", exposure, "R7: preserve the dispute record"),
                _recommendation("WARN_CUSTOMER", exposure, "R7: explain the recurring charge"),
                _recommendation(
                    "CLOSE_NO_FRAUD", exposure, "R7: verification settled the alert"
                ),
            ]
        return [
            _recommendation("CREATE_CASE", exposure, "R7: preserve the disputed charge"),
            _recommendation(
                "VERIFY_WITH_CUSTOMER", exposure, "R7: verify the recurring charge"
            ),
            _recommendation("WARN_CUSTOMER", exposure, "R7: explain the recurring pattern"),
        ]
    if response and ("confirmed" in response.lower() or "passed" in response.lower()):
        return [
            _recommendation(
                "ALLOW_TRANSACTION", exposure, "R3: customer or authentication confirmed activity"
            ),
            _recommendation("CLOSE_NO_FRAUD", exposure, "R3: verification settled the alert"),
        ]
    if response and "no response" in response.lower():
        actions = [
            _recommendation("MONITOR_CARD", exposure, "R4: no response within 24 hours"),
            _recommendation("DECLINE_TRANSACTION", exposure, "R4: decline pending authorization"),
            _recommendation("CREATE_CASE", exposure, "R3a: evidence was requested"),
        ]
        if exposure > 500:
            actions.append(
                _recommendation(
                    "ESCALATE_TO_ANALYST", exposure, "R8: uncertainty with exposure over $500"
                )
            )
        return actions
    if response and ("denied" in response.lower() or "failed" in response.lower()):
        return _fraud_actions(exposure, connected)
    if trigger.trigger_type == "customer_report" or probability >= 0.85:
        return _fraud_actions(exposure, connected)
    if request:
        request_action = (
            "STEP_UP_AUTH" if request.type == "step_up_auth" else "VERIFY_WITH_CUSTOMER"
        )
        return [
            _recommendation(request_action, exposure, "R1: verify a weak signal before blocking"),
            _recommendation("CREATE_CASE", exposure, "R3a: requesting evidence requires a case"),
        ]
    return [
        _recommendation(
            "ALLOW_TRANSACTION", exposure, "R1/R3: evidence supports legitimate activity"
        ),
        _recommendation("CLOSE_NO_FRAUD", exposure, "R3: close the alert as legitimate"),
    ]


def _sar(
    trigger: CaseTrigger,
    flagged: dict[str, Any],
    detection: PatternDetection,
    exposure: float,
    connected: list[str],
    actions: list[RecommendedAction],
) -> SarRecord:
    should_file = any(item.action == "FILE_REPORT" for item in actions)
    if not should_file:
        return SarRecord(
            file=False,
            reason="R3a: this case does not meet the suspicious activity reporting threshold.",
            narrative="",
            subjects=[],
            total_amount_usd=0,
            activity_dates=[],
        )
    date = str(flagged["ts"])[:10]
    subjects = [trigger.customer_id, trigger.card_id, *connected]
    device = flagged.get("device_profile_id")
    if device:
        subjects.append(str(device))
    narrative = (
        f"Customer {trigger.customer_id} and card {trigger.card_id} were investigated after {trigger.trigger_type.replace('_', ' ')} activity. "
        f"Transaction {trigger.flagged_txn_id} occurred on {date} for ${float(flagged.get('TransactionAmt') or 0):.2f} through the {flagged.get('channel', 'unknown')} channel. "
        f"The investigation identified {detection.pattern.replace('_', ' ')} based on graph-linked transaction behavior. "
        f"The suspicious episode totals ${exposure:.2f} and includes {len(detection.affected_txn_ids)} transaction(s). "
        f"Connected cards or shared origins were identified as {', '.join(connected) if connected else 'none beyond the subject card'}. "
        "The bank created a fraud case, restricted the affected card, and escalated the regulatory filing for approval."
    )
    return SarRecord(
        file=True,
        reason="R2/R6/R9: confirmed or strongly suspected fraud meets a reporting condition.",
        narrative=narrative,
        subjects=list(dict.fromkeys(subjects)),
        total_amount_usd=round(exposure, 2),
        activity_dates=[date, date],
    )


class InvestigationAgent:
    def __init__(
        self,
        store: GraphStore,
        narrator: Narrator | None = None,
        probability_model: ProbabilityModel | None = None,
    ) -> None:
        self.store = store
        self.narrator = narrator
        self.probability_model = probability_model
        self.embedder = HashEmbedding()
        builder = StateGraph(InvestigationState)
        builder.add_node("investigate", self._investigation_node)
        builder.add_edge(START, "investigate")
        builder.add_edge("investigate", END)
        self.workflow = builder.compile()

    async def investigate(self, trigger: CaseTrigger) -> InvestigationAnswer:
        result = await self.workflow.ainvoke({"trigger": trigger, "answer": None})
        answer = result["answer"]
        if answer is None:
            raise RuntimeError("investigation workflow completed without an answer")
        return cast(InvestigationAnswer, answer)

    async def _investigation_node(
        self, state: InvestigationState
    ) -> dict[str, InvestigationAnswer]:
        return {"answer": await self._investigate_core(state["trigger"])}

    async def _investigate_core(self, trigger: CaseTrigger) -> InvestigationAnswer:
        started = time.perf_counter()
        starting_calls = self.store.tool_calls
        as_of = trigger.opened_at.isoformat(sep=" ")
        flagged = await self.store.run_query(
            "transaction_context", {"txn_id": trigger.flagged_txn_id, "as_of": as_of}
        )
        if not isinstance(flagged, dict) or not flagged:
            raise LookupError(f"transaction {trigger.flagged_txn_id} was not found")
        history = await self.store.run_query(
            "card_history",
            {"card_id": trigger.card_id, "txn_id": trigger.flagged_txn_id, "as_of": as_of},
        )
        if not isinstance(history, list):
            raise TypeError("card_history query must return a list")
        prior_cases = await self.store.run_query(
            "prior_cases", {"card_id": trigger.card_id, "as_of": as_of, "limit_n": 5}
        )
        connected: list[str] = []
        if flagged.get("device_profile_id"):
            neighbors = await self.store.run_query(
                "device_neighbors",
                {
                    "device_id": flagged["device_profile_id"],
                    "window_start": (trigger.opened_at - timedelta(hours=48)).isoformat(
                        sep=" "
                    ),
                    "as_of": as_of,
                },
            )
            if isinstance(neighbors, dict):
                connected = [
                    str(card)
                    for card in neighbors.get("confirmed_fraud_card_ids", [])
                    if str(card) != trigger.card_id
                ]
        knowledge = await self.store.search_knowledge(
            self.embedder.embed(f"{trigger.trigger_text} fraud policy and similar cases"), top_k=5
        )

        detection = detect_pattern(history, trigger.flagged_txn_id, flagged)
        recurring_ids = recurring_charge_matches(history, trigger.flagged_txn_id)
        recurring_dispute = trigger.trigger_type == "customer_report" and bool(
            recurring_ids
        )
        initial_probability = _probability(trigger, flagged, detection, self.probability_model)
        if recurring_dispute:
            initial_probability = min(initial_probability, 0.45)
        affected = detection.affected_txn_ids if initial_probability >= 0.30 else []
        amount_by_id = {
            str(row["TransactionID"]): abs(float(row.get("TransactionAmt") or 0)) for row in history
        }
        exposure = round(sum(amount_by_id.get(txn_id, 0) for txn_id in affected), 2)

        request: EvidenceRequest | None = None
        response: str | None = None
        if (
            trigger.trigger_type != "customer_report" or recurring_dispute
        ) and 0.15 < initial_probability < 0.85:
            if recurring_dispute:
                request_type = "customer_validation"
            elif trigger.trigger_type == "analyst_request":
                request_type = "analyst_info"
            else:
                request_type = (
                    "step_up_auth" if flagged.get("channel") == "online" else "customer_validation"
                )
            response = simulate_evidence_response(request_type, initial_probability)
            request = EvidenceRequest(
                type=request_type,  # type: ignore[arg-type]
                asked_after_step=4,
                assumed_response=response,
            )
        initial_actions = _actions(
            trigger,
            initial_probability,
            exposure,
            connected,
            request,
            response=None,
            recurring_dispute=recurring_dispute,
        )
        final_probability = _updated_probability(initial_probability, response or "")
        final_actions = _actions(
            trigger,
            final_probability,
            exposure,
            connected,
            request,
            response=response,
            recurring_dispute=recurring_dispute,
        )

        response_text = (response or "").lower()
        settled_fraud = (
            (trigger.trigger_type == "customer_report" and not recurring_dispute)
            or "denied" in response_text
            or "failed" in response_text
            or final_probability >= 0.85
        )
        settled_legitimate = (
            "confirmed" in response_text or "passed" in response_text or final_probability <= 0.30
        )
        if settled_fraud:
            verdict, status = "fraud", "closed_fraud"
            if not affected:
                affected = [trigger.flagged_txn_id]
                exposure = abs(float(flagged.get("TransactionAmt") or 0))
        elif settled_legitimate:
            verdict, status = "legitimate", "closed_legitimate"
            affected, exposure = [], 0.0
            detection = PatternDetection("none", final_probability, [], detection.claim)
        else:
            verdict = "uncertain"
            status = "escalated" if exposure > 500 else "open"

        evidence = [
            EvidenceItem(
                claim=(
                    f"Flagged transaction {trigger.flagged_txn_id} was ${float(flagged.get('TransactionAmt') or 0):.2f}, "
                    f"{flagged.get('channel', 'unknown')}, risk score {float(flagged.get('risk_score') or 0):.2f}."
                ),
                source="graph",
                ref="query:transaction_context",
                entity_ids=[trigger.flagged_txn_id],
            ),
            EvidenceItem(
                claim=detection.claim,
                source="graph",
                ref=f"detector:{detection.pattern}",
                entity_ids=detection.affected_txn_ids,
            ),
        ]
        if trigger.trigger_type == "customer_report":
            evidence.append(
                EvidenceItem(
                    claim="The customer reported that the flagged transaction was unauthorized.",
                    source="customer",
                    ref="trigger:customer_report",
                    entity_ids=[trigger.customer_id, trigger.card_id, trigger.flagged_txn_id],
                )
            )
        if recurring_dispute:
            evidence.append(
                EvidenceItem(
                    claim=(
                        "The disputed charge matches the card's monthly amount, "
                        "product, and channel pattern."
                    ),
                    source="graph",
                    ref="detector:recurring_charge",
                    entity_ids=recurring_ids,
                )
            )
        if request:
            evidence.append(
                EvidenceItem(
                    claim=request.assumed_response,
                    source="customer" if request.type != "analyst_info" else "external",
                    ref="evidence_request:1",
                    entity_ids=[],
                )
            )
        if knowledge:
            evidence.append(
                EvidenceItem(
                    claim="Policy and prior-case guidance was retrieved through GraphRAG.",
                    source="document",
                    ref="vector:knowledge_top_5",
                    entity_ids=[],
                )
            )

        prior_ids = (
            [
                str(item.get("case_id"))
                for item in prior_cases
                if isinstance(item, dict) and item.get("case_id")
            ]
            if isinstance(prior_cases, list)
            else []
        )
        summary = (
            f"The investigation reviewed transaction {trigger.flagged_txn_id}, the card's as-of history, and graph-linked prior cases. "
            f"The evidence supports a {verdict} verdict with probability {final_probability:.2f} and pattern {detection.pattern}. "
            f"Policy controls produced {len(final_actions)} final action(s)."
        )
        sar = _sar(trigger, flagged, detection, exposure, connected, final_actions)
        answer = InvestigationAnswer(
            case_id=trigger.case_id,
            case=CaseRecord(
                status=status,  # type: ignore[arg-type]
                verdict=verdict,  # type: ignore[arg-type]
                fraud_probability=round(final_probability, 4),
                pattern=detection.pattern,
                pattern_description="",
                affected_txn_ids=affected,
                first_suspicious_txn_id=affected[0] if affected else "",
                connected_card_ids=connected,
                connected_device_profiles=(
                    [str(flagged["device_profile_id"])] if flagged.get("device_profile_id") else []
                ),
                exposure_usd=round(exposure, 2),
                evidence=evidence,
                similar_prior_cases=prior_ids,
                summary=summary,
                written_to_graph=False,
                graph_case_id="",
            ),
            evidence_requests=[request] if request else [],
            next_best_actions=NextBestActions(
                initial=initial_actions,
                final=final_actions,
                what_changed=(
                    "nothing"
                    if not request
                    else f"The assumed response changed probability from {initial_probability:.2f} to {final_probability:.2f}."
                ),
            ),
            sar=sar,
            stop_reason=(
                "Verification settled the question."
                if response and "no response" not in response.lower()
                else "The available independent evidence supports a defensible policy action."
            ),
            tool_calls=0,
            tokens=0,
            latency_s=0,
        )
        if self.narrator:
            rewritten_summary, rewritten_sar, tokens = await self.narrator.rewrite(answer)
            answer.case.summary = rewritten_summary
            if answer.sar.file and rewritten_sar:
                answer.sar.narrative = rewritten_sar
            answer.tokens = tokens
        policy_errors = validate_policy_consistency(answer)
        if policy_errors:
            raise ValueError("; ".join(policy_errors))
        graph_case_id = await self.store.write_case(
            answer, card_id=trigger.card_id, opened_at=trigger.opened_at
        )
        answer.case.written_to_graph = self.store.writes_to_tigergraph
        answer.case.graph_case_id = graph_case_id if self.store.writes_to_tigergraph else ""
        answer.tool_calls = self.store.tool_calls - starting_calls
        answer.latency_s = round(time.perf_counter() - started, 4)
        return InvestigationAnswer.model_validate(answer.model_dump())


async def run_batch(
    triggers: list[CaseTrigger], agent: InvestigationAgent, cases_dir: Path
) -> list[InvestigationAnswer]:
    cases_dir.mkdir(parents=True, exist_ok=True)
    answers: list[InvestigationAnswer] = []
    for trigger in sorted(triggers, key=lambda item: (item.opened_at, item.case_id)):
        answer = await agent.investigate(trigger)
        (cases_dir / f"{answer.case_id}.json").write_text(
            answer.model_dump_json(indent=2), encoding="utf-8"
        )
        answers.append(answer)
    return answers
