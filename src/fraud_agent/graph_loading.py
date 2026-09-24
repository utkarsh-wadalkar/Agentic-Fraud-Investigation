"""Prepare compact, normalized CSV files for TigerGraph bulk loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

GRAPH_LOAD_FILES = (
    "customers.csv",
    "cards.csv",
    "transactions.csv",
    "devices.csv",
    "transaction_devices.csv",
    "email_domains.csv",
    "purchaser_emails.csv",
    "recipient_emails.csv",
    "billing_regions.csv",
    "transaction_regions.csv",
    "next_transactions.csv",
    "closed_cases.csv",
    "case_transactions.csv",
    "case_connected_cards.csv",
)


@dataclass(frozen=True)
class GraphLoadManifest:
    directory: Path
    counts: dict[str, int]


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def _export_query(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    query: str,
) -> int:
    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        while rows := cursor.fetchmany(10_000):
            writer.writerows([_csv_value(value) for value in row] for row in rows)
            count += len(rows)
    return count


def _device_expression(alias: str = "i") -> str:
    components = (
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33",
        "DeviceType",
    )
    normalized = ", ".join(
        f"lower(trim(coalesce({alias}.\"{component}\", '')))" for component in components
    )
    return f"'DEV-' || upper(substr(sha256(concat_ws('|', {normalized})), 1, 16))"


def prepare_graph_files(database_path: Path, output_dir: Path) -> GraphLoadManifest:
    """Export one headered CSV per graph vertex/edge family from canonical DuckDB data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_id = _device_expression()
    device_present = " OR ".join(
        f"trim(coalesce(i.\"{field}\", '')) <> ''"
        for field in ("DeviceInfo", "id_30", "id_31", "id_33", "DeviceType")
    )
    queries = {
        "customers.csv": """
            SELECT DISTINCT customer_id
            FROM transactions
            WHERE trim(coalesce(customer_id, '')) <> ''
            ORDER BY customer_id
        """,
        "cards.csv": """
            SELECT card_id, min(coalesce(card4, '')) AS network,
                   min(coalesce(card6, '')) AS card_type,
                   min(customer_id) AS customer_id
            FROM transactions
            WHERE trim(coalesce(card_id, '')) <> ''
            GROUP BY card_id
            ORDER BY card_id
        """,
        "transactions.csv": """
            SELECT t.\"TransactionID\" AS txn_id, t.ts,
                   coalesce(try_cast(t.\"TransactionAmt\" AS DOUBLE), 0.0) AS amount,
                   coalesce(t.\"ProductCD\", '') AS product_cd,
                   coalesce(t.channel, '') AS channel,
                   coalesce(try_cast(t.risk_score AS DOUBLE), 0.0) AS risk_score,
                   coalesce(t.addr1, '') AS region,
                   coalesce(lower(i.id_15) = 'new', false) AS device_new,
                   coalesce(trim(i.id_23) <> '', false) AS proxy_present,
                   coalesce(i.id_34, '') AS match_status,
                   t.card_id
            FROM transactions t
            LEFT JOIN identity i USING (\"TransactionID\")
            ORDER BY t.\"TransactionID\"
        """,
        "devices.csv": f"""
            SELECT DISTINCT {device_id} AS device_id,
                   coalesce(i.\"DeviceInfo\", '') AS label,
                   coalesce(i.\"DeviceType\", '') AS device_type,
                   coalesce(i.id_30, '') AS os,
                   coalesce(i.id_31, '') AS browser,
                   coalesce(i.id_33, '') AS screen
            FROM identity i
            WHERE {device_present}
            ORDER BY device_id
        """,
        "transaction_devices.csv": f"""
            SELECT i.\"TransactionID\" AS txn_id, {device_id} AS device_id
            FROM identity i
            WHERE {device_present}
            ORDER BY txn_id
        """,
        "email_domains.csv": """
            SELECT domain FROM (
                SELECT DISTINCT trim(\"P_emaildomain\") AS domain
                FROM transactions WHERE trim(coalesce(\"P_emaildomain\", '')) <> ''
                UNION
                SELECT DISTINCT trim(\"R_emaildomain\") AS domain
                FROM transactions WHERE trim(coalesce(\"R_emaildomain\", '')) <> ''
            ) ORDER BY domain
        """,
        "purchaser_emails.csv": """
            SELECT \"TransactionID\" AS txn_id, trim(\"P_emaildomain\") AS domain
            FROM transactions
            WHERE trim(coalesce(\"P_emaildomain\", '')) <> ''
            ORDER BY txn_id
        """,
        "recipient_emails.csv": """
            SELECT \"TransactionID\" AS txn_id, trim(\"R_emaildomain\") AS domain
            FROM transactions
            WHERE trim(coalesce(\"R_emaildomain\", '')) <> ''
            ORDER BY txn_id
        """,
        "billing_regions.csv": """
            SELECT DISTINCT trim(addr1) AS region_id
            FROM transactions
            WHERE trim(coalesce(addr1, '')) <> ''
            ORDER BY region_id
        """,
        "transaction_regions.csv": """
            SELECT \"TransactionID\" AS txn_id, trim(addr1) AS region_id
            FROM transactions
            WHERE trim(coalesce(addr1, '')) <> ''
            ORDER BY txn_id
        """,
        "next_transactions.csv": """
            SELECT from_txn_id, to_txn_id FROM (
                SELECT \"TransactionID\" AS from_txn_id,
                       lead(\"TransactionID\") OVER (
                           PARTITION BY card_id ORDER BY ts, \"TransactionID\"
                       ) AS to_txn_id
                FROM transactions
            )
            WHERE to_txn_id IS NOT NULL
            ORDER BY from_txn_id
        """,
        "closed_cases.csv": """
            SELECT case_id, opened_at, 'closed' AS status,
                   CASE WHEN outcome = 'confirmed_fraud' THEN 'fraud'
                        WHEN outcome = 'cleared' THEN 'legitimate'
                        ELSE 'uncertain' END AS verdict,
                   CASE WHEN outcome = 'confirmed_fraud' THEN 1.0
                        WHEN outcome = 'cleared' THEN 0.0
                        ELSE 0.5 END AS fraud_probability,
                   coalesce(pattern, '') AS pattern,
                   coalesce(try_cast(exposure_usd AS DOUBLE), 0.0) AS exposure_usd,
                   coalesce(analyst_notes, '') AS summary,
                   'history' AS source,
                   card_id
            FROM closed_cases
            ORDER BY case_id
        """,
        "case_transactions.csv": """
            SELECT case_id, trim(txn_id) AS txn_id
            FROM closed_cases,
                 unnest(string_split(coalesce(txn_ids, ''), '|')) AS ids(txn_id)
            WHERE trim(txn_id) <> ''
            ORDER BY case_id, txn_id
        """,
        "case_connected_cards.csv": """
            SELECT case_id, trim(ids.card_id) AS card_id
            FROM closed_cases,
                 unnest(string_split(coalesce(connected_card_ids, ''), '|')) AS ids(card_id)
            WHERE trim(ids.card_id) <> ''
            ORDER BY case_id, ids.card_id
        """,
    }
    counts: dict[str, int] = {}
    with duckdb.connect(str(database_path), read_only=True) as connection:
        for filename in GRAPH_LOAD_FILES:
            counts[filename] = _export_query(
                connection,
                output_dir / filename,
                queries[filename],
            )
    return GraphLoadManifest(directory=output_dir, counts=counts)
