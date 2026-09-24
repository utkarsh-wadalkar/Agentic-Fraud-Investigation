"""Streaming source-data preparation and explainable feature extraction."""

from __future__ import annotations

import csv
import hashlib
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

Row = Mapping[str, str]
CARD_FIELDS = tuple(f"card{index}" for index in range(1, 7))
DEVICE_FIELDS = ("DeviceInfo", "id_30", "id_31", "id_33", "DeviceType")


@dataclass(frozen=True)
class DatasetProfile:
    transactions: int
    identities: int
    closed_cases: int
    benchmark_cases: int


@dataclass(frozen=True)
class TransactionFeatures:
    transaction_id: str
    history_count: int
    amount_ratio: float
    region_novelty: float
    channel_novelty: float
    velocity_1h: int
    device_new: float
    proxy_present: float
    risk_score: float


def card_signature(row: Row) -> tuple[str, ...]:
    return tuple((row.get(field) or "").strip().lower() for field in CARD_FIELDS)


def assign_card_ids(
    transactions: Sequence[Row], authoritative_by_txn: Mapping[str, str]
) -> dict[str, str]:
    by_customer: dict[str, dict[tuple[str, ...], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in transactions:
        by_customer[row["customer_id"]][card_signature(row)].append(row["TransactionID"])

    result: dict[str, str] = {}
    for customer_id, signature_txns in by_customer.items():
        labels: dict[tuple[str, ...], str] = {}
        for signature, transaction_ids in signature_txns.items():
            known = {
                authoritative_by_txn[txn_id]
                for txn_id in transaction_ids
                if txn_id in authoritative_by_txn
            }
            if len(known) > 1:
                raise ValueError(f"conflicting authoritative card IDs for {customer_id}: {known}")
            if known:
                label = known.pop()
                if not label.startswith(f"{customer_id}-K"):
                    raise ValueError(f"card {label} does not belong to customer {customer_id}")
                labels[signature] = label

        ordered = sorted(
            signature_txns,
            key=lambda value: (sum(bool(part) for part in value), value),
        )
        used_numbers = {
            int(label.rsplit("-K", 1)[1])
            for label in labels.values()
            if label.rsplit("-K", 1)[1].isdigit()
        }
        next_number = 1
        for signature in ordered:
            if signature not in labels:
                while next_number in used_numbers:
                    next_number += 1
                labels[signature] = f"{customer_id}-K{next_number}"
                used_numbers.add(next_number)
            for transaction_id in signature_txns[signature]:
                result[transaction_id] = authoritative_by_txn.get(transaction_id, labels[signature])
    return result


def device_profile_id(identity: Row) -> str | None:
    normalized = [str(identity.get(field, "") or "").strip().lower() for field in DEVICE_FIELDS]
    if not any(normalized):
        return None
    digest = hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()[:16]
    return f"DEV-{digest.upper()}"


def rows_as_of(rows: Iterable[Row], opened_at: datetime) -> list[Row]:
    return [row for row in rows if datetime.fromisoformat(row["ts"]) <= opened_at]


def extract_transaction_features(
    rows: Sequence[Row], flagged_transaction_id: str, identity: Row | None = None
) -> TransactionFeatures:
    try:
        flagged = next(row for row in rows if row["TransactionID"] == flagged_transaction_id)
    except StopIteration as error:
        raise KeyError(f"unknown transaction {flagged_transaction_id}") from error

    flagged_at = datetime.fromisoformat(flagged["ts"])
    history = [row for row in rows if datetime.fromisoformat(row["ts"]) < flagged_at]
    prior_amounts = [abs(float(row.get("TransactionAmt") or 0)) for row in history]
    baseline = statistics.mean(prior_amounts) if prior_amounts else 0.0
    amount = abs(float(flagged.get("TransactionAmt") or 0))
    amount_ratio = amount / baseline if baseline > 0 else (1.0 if amount == 0 else amount)
    region = flagged.get("addr1") or ""
    channel = flagged.get("channel") or ""
    one_hour = 3600
    velocity = sum(
        0 <= (flagged_at - datetime.fromisoformat(row["ts"])).total_seconds() <= one_hour
        for row in history
    )
    identity = identity or {}
    return TransactionFeatures(
        transaction_id=flagged_transaction_id,
        history_count=len(history),
        amount_ratio=amount_ratio,
        region_novelty=float(
            bool(region) and all((row.get("addr1") or "") != region for row in history)
        ),
        channel_novelty=float(
            bool(channel) and all((row.get("channel") or "") != channel for row in history)
        ),
        velocity_1h=velocity,
        device_new=float((identity.get("id_15") or "").lower() == "new"),
        proxy_present=float(bool(identity.get("id_23"))),
        risk_score=float(flagged.get("risk_score") or 0),
    )


def _count_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def profile_source_dir(source_dir: Path) -> DatasetProfile:
    required = {
        "transactions": source_dir / "transactions.csv",
        "identities": source_dir / "identity.csv",
        "closed_cases": source_dir / "closed_cases_history.csv",
        "benchmark_cases": source_dir / "case_pack.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {', '.join(missing)}")
    return DatasetProfile(**{name: _count_csv(path) for name, path in required.items()})


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _authoritative_cards(source_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with (source_dir / "closed_cases_history.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for transaction_id in filter(None, (row.get("txn_ids") or "").split("|")):
                mapping[transaction_id] = row["card_id"]
    with (source_dir / "case_pack.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("flagged_txn_id") and row.get("card_id"):
                mapping[row["flagged_txn_id"]] = row["card_id"]
    return mapping


def prepare_duckdb(source_dir: Path, database_path: Path) -> Path:
    """Create a canonical local analytical store without altering source files."""
    source_dir = source_dir.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    authoritative = _authoritative_cards(source_dir)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("DROP TABLE IF EXISTS transactions")
        connection.execute("DROP TABLE IF EXISTS identity")
        connection.execute("DROP TABLE IF EXISTS closed_cases")
        connection.execute("DROP TABLE IF EXISTS case_pack")
        transactions_path = _sql_path(source_dir / "transactions.csv")
        connection.execute(
            "CREATE TEMP TABLE raw_transactions AS "
            f"SELECT * FROM read_csv_auto('{transactions_path}', all_varchar=true, header=true)"
        )
        connection.execute("CREATE TEMP TABLE authority(txn_id VARCHAR, card_id VARCHAR)")
        if authoritative:
            connection.executemany(
                "INSERT INTO authority VALUES (?, ?)", list(authoritative.items())
            )
        signature_parts = ", ".join(f"coalesce(\"{field}\", '')" for field in CARD_FIELDS)
        completeness = " + ".join(
            f"CASE WHEN coalesce(\"{field}\", '') <> '' THEN 1 ELSE 0 END" for field in CARD_FIELDS
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE transaction_signatures AS
            SELECT *, concat_ws('|', {signature_parts}) AS card_signature,
                   ({completeness}) AS card_completeness
            FROM raw_transactions
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE ranked_signatures AS
            SELECT customer_id, card_signature,
                   row_number() OVER (
                       PARTITION BY customer_id
                       ORDER BY min(card_completeness), card_signature
                   ) AS fallback_number
            FROM transaction_signatures
            GROUP BY customer_id, card_signature
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE signature_labels AS
            SELECT t.customer_id, t.card_signature, min(a.card_id) AS card_id
            FROM transaction_signatures t
            JOIN authority a ON a.txn_id = t."TransactionID"
            GROUP BY t.customer_id, t.card_signature
            """
        )
        connection.execute(
            """
            CREATE TABLE transactions AS
            SELECT t.* EXCLUDE(card_signature, card_completeness),
                   coalesce(a.card_id, s.card_id,
                       t.customer_id || '-K' || cast(r.fallback_number AS VARCHAR)
                   ) AS card_id
            FROM transaction_signatures t
            JOIN ranked_signatures r USING (customer_id, card_signature)
            LEFT JOIN authority a ON a.txn_id = t."TransactionID"
            LEFT JOIN signature_labels s USING (customer_id, card_signature)
            """
        )
        for table_name, filename in (
            ("identity", "identity.csv"),
            ("closed_cases", "closed_cases_history.csv"),
            ("case_pack", "case_pack.csv"),
        ):
            csv_path = _sql_path(source_dir / filename)
            connection.execute(
                f"CREATE TABLE {table_name} AS "
                f"SELECT * FROM read_csv_auto('{csv_path}', all_varchar=true, header=true)"
            )
        connection.execute('CREATE INDEX transactions_id_idx ON transactions("TransactionID")')
        connection.execute(
            "CREATE INDEX transactions_customer_ts_idx ON transactions(customer_id, ts)"
        )
        connection.execute("CREATE INDEX transactions_card_ts_idx ON transactions(card_id, ts)")
    return database_path
