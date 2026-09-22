"""Chronological model-training primitives for fraud probability."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_agent.data import TransactionFeatures


def chronological_split[T](
    rows: Sequence[tuple[datetime, T]], test_fraction: float = 0.2
) -> tuple[list[tuple[datetime, T]], list[tuple[datetime, T]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = sorted(rows, key=lambda item: item[0])
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    split_at = max(0, len(ordered) - test_size)
    return ordered[:split_at], ordered[split_at:]


def feature_vector(features: TransactionFeatures) -> list[float]:
    return [
        float(features.history_count),
        features.amount_ratio,
        features.region_novelty,
        features.channel_novelty,
        float(features.velocity_1h),
        features.device_new,
        features.proxy_present,
        features.risk_score,
    ]


class FraudProbabilityModel:
    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
                ),
            ]
        )
        self._fitted = False

    def fit(
        self, samples: Sequence[TransactionFeatures], labels: Sequence[int]
    ) -> FraudProbabilityModel:
        if len(samples) != len(labels) or not samples:
            raise ValueError("samples and labels must be non-empty and have equal length")
        if len(set(labels)) != 2:
            raise ValueError("training data must contain fraud and cleared examples")
        matrix = np.asarray([feature_vector(sample) for sample in samples], dtype=float)
        self._pipeline.fit(matrix, np.asarray(labels, dtype=int))
        self._fitted = True
        return self

    def predict_probability(self, features: TransactionFeatures) -> float:
        if not self._fitted:
            raise RuntimeError("model must be fitted before prediction")
        probability = self._pipeline.predict_proba(
            np.asarray([feature_vector(features)], dtype=float)
        )[0, 1]
        return float(min(0.99, max(0.01, probability)))
