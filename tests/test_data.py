from __future__ import annotations

import csv
import importlib
from datetime import datetime


def module():
    return importlib.import_module("fraud_agent.data")


def test_card_mapping_prefers_authoritative_label_and_assigns_blank_signature_first() -> None:
    rows = [
        {
            "TransactionID": "1",
            "customer_id": "C00001",
            "card1": "123",
            "card2": "",
            "card3": "",
            "card4": "",
            "card5": "",
            "card6": "",
        },
        {
            "TransactionID": "2",
            "customer_id": "C00001",
            "card1": "123",
            "card2": "111",
            "card3": "150",
            "card4": "visa",
            "card5": "226",
            "card6": "debit",
        },
        {
            "TransactionID": "3",
            "customer_id": "C00002",
            "card1": "999",
            "card2": "222",
            "card3": "150",
            "card4": "mastercard",
            "card5": "117",
            "card6": "credit",
        },
    ]

    mapping = module().assign_card_ids(rows, {"2": "C00001-K2"})

    assert mapping == {"1": "C00001-K1", "2": "C00001-K2", "3": "C00002-K1"}


def test_device_profile_is_stable_and_skips_empty_identity() -> None:
    identity = {
        "DeviceInfo": "Windows",
        "id_30": "Windows 10",
        "id_31": "edge 16.0",
        "id_33": "1366x768",
        "DeviceType": "desktop",
    }

    first = module().device_profile_id(identity)
    second = module().device_profile_id(dict(reversed(list(identity.items()))))

    assert first == second
    assert first.startswith("DEV-")
    assert module().device_profile_id({key: "" for key in identity}) is None


def test_rows_as_of_excludes_future_transactions() -> None:
    rows = [
        {"TransactionID": "1", "ts": "2016-11-01 10:00:00"},
        {"TransactionID": "2", "ts": "2016-11-02 10:00:00"},
        {"TransactionID": "3", "ts": "2016-11-03 10:00:00"},
    ]

    result = module().rows_as_of(rows, datetime(2016, 11, 2, 10, 0, 0))

    assert [row["TransactionID"] for row in result] == ["1", "2"]


def test_extract_features_uses_only_prior_history() -> None:
    rows = [
        {
            "TransactionID": "1",
            "TransactionAmt": "10",
            "ts": "2016-11-01 09:00:00",
            "channel": "in_person",
            "addr1": "100",
        },
        {
            "TransactionID": "2",
            "TransactionAmt": "12",
            "ts": "2016-11-01 10:00:00",
            "channel": "in_person",
            "addr1": "100",
        },
        {
            "TransactionID": "3",
            "TransactionAmt": "100",
            "ts": "2016-11-01 11:00:00",
            "channel": "online",
            "addr1": "999",
            "risk_score": "0.8",
        },
        {
            "TransactionID": "4",
            "TransactionAmt": "9999",
            "ts": "2016-11-01 12:00:00",
            "channel": "online",
            "addr1": "999",
        },
    ]

    features = module().extract_transaction_features(rows, "3", {"id_15": "New", "id_23": ""})

    assert features.history_count == 2
    assert round(features.amount_ratio, 4) == 9.0909
    assert features.region_novelty == 1.0
    assert features.channel_novelty == 1.0
    assert features.device_new == 1.0


def write_csv(path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_profile_source_dir_counts_all_required_files(tmp_path) -> None:
    write_csv(
        tmp_path / "transactions.csv",
        ["TransactionID", "customer_id"],
        [{"TransactionID": "1", "customer_id": "C1"}, {"TransactionID": "2", "customer_id": "C2"}],
    )
    write_csv(tmp_path / "identity.csv", ["TransactionID"], [{"TransactionID": "1"}])
    write_csv(tmp_path / "closed_cases_history.csv", ["case_id"], [{"case_id": "CC-1"}])
    write_csv(tmp_path / "case_pack.csv", ["case_id"], [{"case_id": "HHG-1"}])

    profile = module().profile_source_dir(tmp_path)

    assert profile.transactions == 2
    assert profile.identities == 1
    assert profile.closed_cases == 1
    assert profile.benchmark_cases == 1


def test_prepare_duckdb_creates_canonical_tables_with_authoritative_card_ids(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    fields = [
        "TransactionID",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "customer_id",
        "ts",
        "channel",
        "risk_score",
    ]
    write_csv(
        source / "transactions.csv",
        fields,
        [
            dict.fromkeys(fields, "")
            | {
                "TransactionID": "1",
                "TransactionAmt": "12.00",
                "card1": "123",
                "customer_id": "C00001",
                "ts": "2016-10-01 00:00:00",
                "channel": "online",
            },
            dict.fromkeys(fields, "")
            | {
                "TransactionID": "2",
                "TransactionAmt": "99.00",
                "card1": "123",
                "card2": "111",
                "card3": "150",
                "card4": "visa",
                "card5": "226",
                "card6": "debit",
                "customer_id": "C00001",
                "ts": "2016-11-01 00:00:00",
                "channel": "online",
                "risk_score": "0.8",
            },
        ],
    )
    write_csv(
        source / "identity.csv",
        ["TransactionID", "DeviceInfo"],
        [{"TransactionID": "2", "DeviceInfo": "Windows"}],
    )
    write_csv(
        source / "closed_cases_history.csv",
        ["case_id", "card_id", "txn_ids"],
        [{"case_id": "CC-1", "card_id": "C00001-K2", "txn_ids": "2"}],
    )
    write_csv(
        source / "case_pack.csv",
        ["case_id", "flagged_txn_id", "card_id"],
        [{"case_id": "HHG-1", "flagged_txn_id": "2", "card_id": "C00001-K2"}],
    )
    database = tmp_path / "prepared.duckdb"

    module().prepare_duckdb(source, database)

    import duckdb

    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(
            'SELECT "TransactionID", card_id FROM transactions ORDER BY "TransactionID"'
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    assert rows == [("1", "C00001-K1"), ("2", "C00001-K2")]
    assert {"transactions", "identity", "closed_cases", "case_pack"} <= tables
