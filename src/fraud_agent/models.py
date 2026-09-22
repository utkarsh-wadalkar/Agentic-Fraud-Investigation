"""Typed benchmark answer contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ActionName = Literal[
    "ALLOW_TRANSACTION",
    "DECLINE_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "BLOCK_CARD",
    "BLOCK_ALL_CARDS",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "FILE_REPORT",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
]
ApprovalRoute = Literal["auto", "L1", "L2"]


class EvidenceItem(BaseModel):
    claim: str
    source: Literal["graph", "document", "customer", "external"]
    ref: str
    entity_ids: list[str]


class EvidenceRequest(BaseModel):
    type: Literal["customer_validation", "step_up_auth", "analyst_info"]
    asked_after_step: int = Field(ge=1)
    assumed_response: str


class RecommendedAction(BaseModel):
    action: ActionName
    route: ApprovalRoute
    reason: str


class NextBestActions(BaseModel):
    initial: list[RecommendedAction]
    final: list[RecommendedAction]
    what_changed: str


class CaseRecord(BaseModel):
    status: Literal["open", "closed_fraud", "closed_legitimate", "escalated"]
    verdict: Literal["fraud", "legitimate", "uncertain"]
    fraud_probability: float = Field(ge=0, le=1)
    pattern: Literal[
        "card_testing",
        "card_not_present_fraud",
        "card_not_present_new_device",
        "out_of_region_use",
        "account_takeover",
        "undocumented",
        "none",
    ]
    pattern_description: str = ""
    affected_txn_ids: list[str]
    first_suspicious_txn_id: str
    connected_card_ids: list[str]
    connected_device_profiles: list[str]
    exposure_usd: float = Field(ge=0)
    evidence: list[EvidenceItem]
    similar_prior_cases: list[str]
    summary: str
    written_to_graph: bool
    graph_case_id: str

    @model_validator(mode="after")
    def validate_case_invariants(self) -> CaseRecord:
        if self.verdict == "legitimate":
            if self.affected_txn_ids or self.exposure_usd != 0:
                raise ValueError("legitimate case must have no affected transactions or exposure")
            if self.pattern != "none":
                raise ValueError("legitimate case must use pattern 'none'")
        if self.pattern == "undocumented" and not self.pattern_description.strip():
            raise ValueError("undocumented pattern requires a description")
        if self.written_to_graph and not self.graph_case_id:
            raise ValueError("written graph case requires graph_case_id")
        return self


class SarRecord(BaseModel):
    file: bool
    reason: str
    narrative: str
    subjects: list[str]
    total_amount_usd: float = Field(ge=0)
    activity_dates: list[str]

    @model_validator(mode="after")
    def validate_sar_shape(self) -> SarRecord:
        if self.file:
            if not self.narrative.strip() or not self.subjects or len(self.activity_dates) != 2:
                raise ValueError("filed SAR requires narrative, subjects, and two activity dates")
        elif self.narrative or self.subjects or self.total_amount_usd or self.activity_dates:
            raise ValueError("unfiled SAR must have empty narrative, subjects, amount, and dates")
        return self


class InvestigationAnswer(BaseModel):
    case_id: str
    case: CaseRecord
    evidence_requests: list[EvidenceRequest]
    next_best_actions: NextBestActions
    sar: SarRecord
    stop_reason: str
    tool_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_answer_invariants(self) -> InvestigationAnswer:
        final_names = {item.action for item in self.next_best_actions.final}
        if self.sar.file != ("FILE_REPORT" in final_names):
            raise ValueError("sar.file must agree with final FILE_REPORT action")
        if not self.evidence_requests:
            if self.next_best_actions.initial != self.next_best_actions.final:
                raise ValueError(
                    "initial and final actions must match when no evidence was requested"
                )
            if self.next_best_actions.what_changed != "nothing":
                raise ValueError("what_changed must be 'nothing' when no evidence was requested")
        return self
