from __future__ import annotations

import csv
import importlib

import duckdb

from fraud_agent.data import device_profile_id


def module():
    return importlib.import_module("fraud_agent.graph_loading")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_graph_files_exports_normalized_vertices_and_edges(tmp_path) -> None:
    database = tmp_path / "fraud.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE transactions(
                "TransactionID" VARCHAR, ts VARCHAR, "TransactionAmt" VARCHAR,
                "ProductCD" VARCHAR, channel VARCHAR, risk_score VARCHAR,
                addr1 VARCHAR, "P_emaildomain" VARCHAR, "R_emaildomain" VARCHAR,
                customer_id VARCHAR, card_id VARCHAR, card4 VARCHAR, card6 VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO transactions VALUES
              ('T1','2016-10-01 10:00:00','10.00','W','in_person','0.1','100',
               'buyer.test','', 'C1','C1-K1','visa','debit'),
              ('T2','2016-10-01 11:00:00','25.50','C','online','0.8','200',
               'buyer.test','recipient.test','C1','C1-K1','visa','debit')
            """
        )
        connection.execute(
            """
            CREATE TABLE identity(
                "TransactionID" VARCHAR, "DeviceInfo" VARCHAR, id_15 VARCHAR,
                id_23 VARCHAR, id_30 VARCHAR, id_31 VARCHAR, id_33 VARCHAR,
                id_34 VARCHAR, "DeviceType" VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO identity VALUES
              ('T2','Pixel','New','IP_PROXY:ANONYMOUS','Android','Chrome','1080x1920',
               'match_status:1','mobile')
            """
        )
        connection.execute(
            """
            CREATE TABLE closed_cases(
                case_id VARCHAR, card_id VARCHAR, opened_at VARCHAR, outcome VARCHAR,
                pattern VARCHAR, exposure_usd VARCHAR, analyst_notes VARCHAR,
                txn_ids VARCHAR, connected_card_ids VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO closed_cases VALUES
              ('CC-1','C1-K1','2016-09-01 09:00:00','confirmed_fraud',
               'card_not_present_fraud','25.50','Known fraud','T1|T2','C2-K1')
            """
        )

    output = tmp_path / "graph-load"
    manifest = module().prepare_graph_files(database, output)

    assert set(manifest.counts) == {
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
    }
    assert manifest.counts["transactions.csv"] == 2
    assert read_rows(output / "customers.csv") == [{"customer_id": "C1"}]
    assert read_rows(output / "cards.csv") == [
        {
            "card_id": "C1-K1",
            "network": "visa",
            "card_type": "debit",
            "customer_id": "C1",
        }
    ]

    device_id = device_profile_id(
        {
            "DeviceInfo": "Pixel",
            "id_30": "Android",
            "id_31": "Chrome",
            "id_33": "1080x1920",
            "DeviceType": "mobile",
        }
    )
    assert device_id is not None
    assert read_rows(output / "devices.csv") == [
        {
            "device_id": device_id,
            "label": "Pixel",
            "device_type": "mobile",
            "os": "Android",
            "browser": "Chrome",
            "screen": "1080x1920",
        }
    ]
    assert read_rows(output / "transaction_devices.csv") == [
        {"txn_id": "T2", "device_id": device_id}
    ]
    transaction = read_rows(output / "transactions.csv")[1]
    assert transaction == {
        "txn_id": "T2",
        "ts": "2016-10-01 11:00:00",
        "amount": "25.5",
        "product_cd": "C",
        "channel": "online",
        "risk_score": "0.8",
        "region": "200",
        "device_new": "true",
        "proxy_present": "true",
        "match_status": "match_status:1",
        "card_id": "C1-K1",
    }
    assert read_rows(output / "next_transactions.csv") == [
        {"from_txn_id": "T1", "to_txn_id": "T2"}
    ]
    assert read_rows(output / "closed_cases.csv") == [
        {
            "case_id": "CC-1",
            "opened_at": "2016-09-01 09:00:00",
            "status": "closed",
            "verdict": "fraud",
            "fraud_probability": "1.0",
            "pattern": "card_not_present_fraud",
            "exposure_usd": "25.5",
            "summary": "Known fraud",
            "source": "history",
            "card_id": "C1-K1",
        }
    ]
    assert read_rows(output / "case_transactions.csv") == [
        {"case_id": "CC-1", "txn_id": "T1"},
        {"case_id": "CC-1", "txn_id": "T2"},
    ]
    assert read_rows(output / "case_connected_cards.csv") == [
        {"case_id": "CC-1", "card_id": "C2-K1"}
    ]


def test_loading_asset_maps_every_prepared_file(tmp_path) -> None:
    graph = importlib.import_module("fraud_agent.graph")

    loading = graph.render_graph_assets("FraudGraph")["loading.gsql"]

    for filename in module().GRAPH_LOAD_FILES:
        assert filename.removesuffix(".csv") + "_file" in loading
    assert "TO EDGE FROM_DEVICE" in loading
    assert "TO EDGE CITES_PRIOR_CASE" not in loading
    assert '$"' not in loading
    assert 'HEADER="false"' in loading
