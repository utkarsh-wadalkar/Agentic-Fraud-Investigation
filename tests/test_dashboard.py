from __future__ import annotations

import json

from tests.test_domain import make_answer


def test_dashboard_view_model_aggregates_cases_and_graph_edges(tmp_path) -> None:
    fraud = make_answer()
    fraud.case_id = "HHG-001"
    fraud.case.connected_card_ids = ["C2-K1"]
    fraud.case.connected_device_profiles = ["DEV-1"]
    legitimate = make_answer()
    legitimate.case_id = "HHG-002"
    legitimate.case.verdict = "legitimate"
    legitimate.case.status = "closed_legitimate"
    legitimate.case.pattern = "none"
    legitimate.case.affected_txn_ids = []
    legitimate.case.first_suspicious_txn_id = ""
    legitimate.case.exposure_usd = 0
    legitimate.sar.file = False
    legitimate.sar.reason = "R3"
    legitimate.sar.narrative = ""
    legitimate.sar.subjects = []
    legitimate.sar.total_amount_usd = 0
    legitimate.sar.activity_dates = []
    legitimate.next_best_actions.final = []
    legitimate.next_best_actions.initial = []
    for answer in (fraud, legitimate):
        (tmp_path / f"{answer.case_id}.json").write_text(answer.model_dump_json(), encoding="utf-8")

    from fraud_agent.dashboard import build_dashboard_view

    view = build_dashboard_view(tmp_path)

    assert view.total_cases == 2
    assert view.verdict_counts == {"fraud": 1, "legitimate": 1, "uncertain": 0}
    assert view.total_exposure == fraud.case.exposure_usd
    assert {edge[1] for edge in view.case_graph["HHG-001"]} == {
        "C2-K1",
        "DEV-1",
    }
    assert json.loads(view.answers["HHG-002"].model_dump_json())["case_id"] == "HHG-002"
