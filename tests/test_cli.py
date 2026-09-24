from __future__ import annotations

import csv
import importlib
from pathlib import Path

from fraud_agent.config import Settings
from tests.test_domain import make_answer


def module():
    return importlib.import_module("fraud_agent.cli")


def test_load_triggers_parses_and_orders_case_pack(tmp_path) -> None:
    path = tmp_path / "case_pack.csv"
    fields = [
        "case_id",
        "opened_at",
        "trigger_type",
        "trigger_text",
        "flagged_txn_id",
        "card_id",
        "customer_id",
        "risk_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "case_id": "HHG-002",
                    "opened_at": "2016-12-01 00:00:00",
                    "trigger_type": "risk_score",
                    "trigger_text": "Review",
                    "flagged_txn_id": "T2",
                    "card_id": "C1-K1",
                    "customer_id": "C1",
                    "risk_score": "0.7",
                },
                {
                    "case_id": "HHG-001",
                    "opened_at": "2016-11-01 00:00:00",
                    "trigger_type": "customer_report",
                    "trigger_text": "Denied",
                    "flagged_txn_id": "T1",
                    "card_id": "C1-K1",
                    "customer_id": "C1",
                    "risk_score": "",
                },
            ]
        )

    triggers = module().load_triggers(path)

    assert [trigger.case_id for trigger in triggers] == ["HHG-001", "HHG-002"]
    assert triggers[1].risk_score == 0.7


def test_validate_answer_directory_reports_valid_json_files(tmp_path) -> None:
    answer = make_answer()
    (tmp_path / "HHG-001.json").write_text(answer.model_dump_json(), encoding="utf-8")

    report = module().validate_answer_directory(tmp_path, expected_count=1)

    assert report.valid_files == 1
    assert report.errors == []


def test_auto_backend_uses_mcp_only_when_tigergraph_is_configured() -> None:
    assert module().resolve_backend(Settings(tg_host="", tg_api_token=""), "auto") == "local"
    ready = Settings(
        tg_host="https://example", tg_graphname="FraudGraph", tg_secret="", tg_api_token="x"
    )
    assert module().resolve_backend(ready, "auto") == "mcp"
    assert module().resolve_backend(ready, "local") == "local"
    secret_ready = Settings(
        tg_host="https://example", tg_graphname="FraudGraph", tg_secret="x"
    )
    assert module().resolve_backend(secret_ready, "auto") == "mcp"


def test_load_or_train_model_persists_new_model(monkeypatch, tmp_path) -> None:
    saved: list[Path] = []

    class FakeModel:
        def save(self, path: Path) -> Path:
            saved.append(path)
            path.write_bytes(b"model")
            return path

    fake_model = FakeModel()
    monkeypatch.setattr(module(), "train_from_duckdb", lambda database: (fake_model, 17))
    model_path = tmp_path / "fraud-model.joblib"

    model, samples = module().load_or_train_model(tmp_path / "fraud.duckdb", model_path)

    assert model is fake_model
    assert samples == 17
    assert saved == [model_path]


def test_graph_pipeline_commands_are_registered() -> None:
    parser = module().build_parser()

    prepare = parser.parse_args(["graph-prepare"])
    deploy = parser.parse_args(["graph-deploy", "--replace"])
    load = parser.parse_args(["graph-load"])
    knowledge = parser.parse_args(["graph-knowledge"])

    assert prepare.database == Path(".artifacts/fraud.duckdb")
    assert prepare.out == Path(".artifacts/graph-load")
    assert deploy.replace is True
    assert load.data_dir == Path(".artifacts/graph-load")
    assert knowledge.source == Path("Drive Files/README.md")
