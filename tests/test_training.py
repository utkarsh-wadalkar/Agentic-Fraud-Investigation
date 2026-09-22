from __future__ import annotations

import importlib
from datetime import datetime, timedelta

from fraud_agent.data import TransactionFeatures


def module():
    return importlib.import_module("fraud_agent.training")


def feature(
    transaction_id: str, risk: float, amount_ratio: float, new_device: float
) -> TransactionFeatures:
    return TransactionFeatures(
        transaction_id=transaction_id,
        history_count=10,
        amount_ratio=amount_ratio,
        region_novelty=new_device,
        channel_novelty=new_device,
        velocity_1h=int(new_device * 3),
        device_new=new_device,
        proxy_present=new_device,
        risk_score=risk,
    )


def test_chronological_split_never_places_future_rows_in_training() -> None:
    start = datetime(2016, 7, 1)
    rows = [(start + timedelta(days=index), index) for index in range(10)]

    train, test = module().chronological_split(rows, test_fraction=0.2)

    assert [value for _, value in train] == list(range(8))
    assert [value for _, value in test] == [8, 9]
    assert max(when for when, _ in train) < min(when for when, _ in test)


def test_fraud_probability_model_scores_strong_signal_above_clear_signal() -> None:
    samples = []
    labels = []
    for index in range(12):
        fraud = index >= 6
        samples.append(
            feature(
                str(index),
                risk=0.85 if fraud else 0.15,
                amount_ratio=8.0 if fraud else 1.0,
                new_device=1.0 if fraud else 0.0,
            )
        )
        labels.append(int(fraud))
    model = module().FraudProbabilityModel().fit(samples, labels)

    clear_probability = model.predict_probability(feature("clear", 0.1, 1.0, 0.0))
    fraud_probability = model.predict_probability(feature("fraud", 0.9, 10.0, 1.0))

    assert 0 <= clear_probability < 0.5
    assert 0.5 < fraud_probability <= 1
