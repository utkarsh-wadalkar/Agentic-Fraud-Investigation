"""Fraud-policy enforcement."""

from __future__ import annotations

from fraud_agent.models import ActionName, ApprovalRoute, InvestigationAnswer


def approval_route(action: ActionName | str, exposure_usd: float) -> ApprovalRoute:
    if action in {"BLOCK_ALL_CARDS", "FILE_REPORT"}:
        return "L2"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure_usd <= 2500 else "L2"
    return "auto"


def validate_policy_consistency(answer: InvestigationAnswer) -> list[str]:
    errors: list[str] = []
    exposure = answer.case.exposure_usd
    for phase in (answer.next_best_actions.initial, answer.next_best_actions.final):
        for recommendation in phase:
            expected = approval_route(recommendation.action, exposure)
            if recommendation.route != expected:
                errors.append(
                    f"{recommendation.action} must use {expected} at exposure {exposure:.2f}"
                )
    if answer.case.fraud_probability >= 0.30 or answer.evidence_requests:
        final_names = {item.action for item in answer.next_best_actions.final}
        if "CREATE_CASE" not in final_names and answer.case.status != "closed_legitimate":
            errors.append("R3a requires CREATE_CASE for probability >= 0.30 or evidence requests")
    return list(dict.fromkeys(errors))
