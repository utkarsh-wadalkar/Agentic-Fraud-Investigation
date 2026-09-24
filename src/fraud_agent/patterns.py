"""Explainable fraud-pattern detectors over as-of transaction history."""

# ruff: noqa: E501 -- evidence claims remain auditable as complete sentences.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

PatternName = Literal[
    "card_testing",
    "card_not_present_fraud",
    "card_not_present_new_device",
    "out_of_region_use",
    "account_takeover",
    "undocumented",
    "none",
]


@dataclass(frozen=True)
class PatternDetection:
    pattern: PatternName
    score: float
    affected_txn_ids: list[str]
    claim: str


def _time(row: Mapping[str, str]) -> datetime:
    return datetime.fromisoformat(row["ts"])


def _amount(row: Mapping[str, str]) -> float:
    return abs(float(row.get("TransactionAmt") or 0))


def recurring_charge_matches(
    transactions: Sequence[Mapping[str, str]], flagged_transaction_id: str
) -> list[str]:
    """Find a monthly amount/product/channel match usable for policy R7."""
    try:
        flagged = next(
            row for row in transactions if row["TransactionID"] == flagged_transaction_id
        )
    except StopIteration as error:
        raise KeyError(f"unknown transaction {flagged_transaction_id}") from error
    flagged_at = _time(flagged)
    flagged_amount = _amount(flagged)
    tolerance = max(1.0, flagged_amount * 0.02)
    matches = [
        row
        for row in transactions
        if row["TransactionID"] != flagged_transaction_id
        and timedelta(days=25) <= flagged_at - _time(row) <= timedelta(days=35)
        and row.get("channel") == flagged.get("channel")
        and row.get("ProductCD") == flagged.get("ProductCD")
        and abs(_amount(row) - flagged_amount) <= tolerance
    ]
    if not matches:
        return []
    closest = min(matches, key=lambda row: abs((flagged_at - _time(row)).days - 30))
    return [str(closest["TransactionID"]), flagged_transaction_id]


def detect_pattern(
    transactions: Sequence[Mapping[str, str]],
    flagged_transaction_id: str,
    identity: Mapping[str, str] | None = None,
) -> PatternDetection:
    ordered = sorted(transactions, key=_time)
    try:
        flagged = next(row for row in ordered if row["TransactionID"] == flagged_transaction_id)
    except StopIteration as error:
        raise KeyError(f"unknown transaction {flagged_transaction_id}") from error
    flagged_at = _time(flagged)
    identity = identity or {}

    testing_window = [
        row
        for row in ordered
        if flagged_at - timedelta(hours=1) <= _time(row) <= flagged_at
        and row.get("channel") == "online"
    ]
    tiny = [row for row in testing_window if _amount(row) < 5 and _time(row) < flagged_at]
    if len(tiny) >= 3 and _amount(flagged) > 100:
        affected = [row["TransactionID"] for row in tiny[-3:]] + [flagged_transaction_id]
        return PatternDetection(
            pattern="card_testing",
            score=0.90,
            affected_txn_ids=affected,
            claim="At least three online authorizations under $5 occurred within one hour before a purchase over $100.",
        )

    prior = [row for row in ordered if _time(row) < flagged_at]
    channel = flagged.get("channel") or ""
    region = flagged.get("addr1") or ""
    new_device = (identity.get("id_15") or "").lower() == "new"
    proxy = bool(identity.get("id_23"))
    mismatch = (identity.get("id_34") or "") in {"match_status:0", "match_status:1"}
    mixed_channel = bool(prior) and any(row.get("channel") != channel for row in prior)

    if channel == "online" and mixed_channel and (proxy or mismatch):
        return PatternDetection(
            pattern="account_takeover",
            score=0.78,
            affected_txn_ids=[flagged_transaction_id],
            claim="Online activity conflicts with prior channel behavior and carries proxy or identity-match anomalies.",
        )
    if channel == "online" and new_device:
        return PatternDetection(
            pattern="card_not_present_new_device",
            score=0.75,
            affected_txn_ids=[flagged_transaction_id],
            claim="The online purchase originated from a device marked New for this account.",
        )
    if channel == "in_person" and region and prior:
        known_regions = {row.get("addr1") for row in prior if row.get("addr1")}
        if region not in known_regions:
            return PatternDetection(
                pattern="out_of_region_use",
                score=0.65,
                affected_txn_ids=[flagged_transaction_id],
                claim=f"The in-person purchase occurred in previously unseen billing region {region}.",
            )
    if channel == "online":
        prior_amounts = [_amount(row) for row in prior]
        baseline = sum(prior_amounts) / len(prior_amounts) if prior_amounts else 0
        if not prior or baseline == 0 or _amount(flagged) >= max(100, baseline * 3):
            return PatternDetection(
                pattern="card_not_present_fraud",
                score=0.60,
                affected_txn_ids=[flagged_transaction_id],
                claim="The online transaction is materially inconsistent with the card's prior amount history.",
            )
    return PatternDetection(
        pattern="none",
        score=0.15,
        affected_txn_ids=[],
        claim="No known fraud pattern was established from the available graph evidence.",
    )
