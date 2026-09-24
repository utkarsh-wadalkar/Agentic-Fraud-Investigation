"""Chronological model-training primitives for fraud probability."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from fraud_agent.data import TransactionFeatures

EXCLUDED_TABULAR_FIELDS = {
    "TransactionID",
    "TransactionDT",
    "ts",
    "customer_id",
    "card_id",
    "case_id",
    "outcome",
}


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


class TabularFraudModel:
    """Sparse mixed-type classifier trained only on supplied closed cases."""

    def __init__(self) -> None:
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = Pipeline(
            [
                ("scale", MaxAbsScaler()),
                (
                    "classifier",
                    LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000),
                ),
            ]
        )
        self.fields: list[str] = []
        self.numeric_fields: set[str] = set()
        self._fitted = False

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None or value == "":
            return True
        try:
            return not isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_number(cls, value: Any) -> bool:
        if cls._is_missing(value):
            return False
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _normalize(self, row: dict[str, Any]) -> dict[str, float | str]:
        normalized: dict[str, float | str] = {}
        for field in self.fields:
            value = row.get(field)
            if field in self.numeric_fields:
                normalized[field] = float(str(value)) if self._is_number(value) else 0.0
                normalized[f"{field}__missing"] = float(self._is_missing(value))
            else:
                normalized[field] = str(value) if not self._is_missing(value) else "__MISSING__"
        return normalized

    def fit_rows(self, rows: Sequence[dict[str, Any]], labels: Sequence[int]) -> TabularFraudModel:
        if len(rows) != len(labels) or not rows:
            raise ValueError("rows and labels must be non-empty and have equal length")
        if len(set(labels)) != 2:
            raise ValueError("training data must contain fraud and cleared examples")
        self.fields = sorted(
            {key for row in rows for key in row if key not in EXCLUDED_TABULAR_FIELDS}
        )
        for field in self.fields:
            present = [row.get(field) for row in rows if not self._is_missing(row.get(field))]
            if present and sum(self._is_number(value) for value in present) / len(present) >= 0.9:
                self.numeric_fields.add(field)
        matrix = self.vectorizer.fit_transform([self._normalize(row) for row in rows])
        self.classifier.fit(matrix, np.asarray(labels, dtype=int))
        self._fitted = True
        return self

    def predict_row(self, row: dict[str, Any]) -> float:
        if not self._fitted:
            raise RuntimeError("model must be fitted before prediction")
        matrix = self.vectorizer.transform([self._normalize(row)])
        raw_probability = self.classifier.predict_proba(matrix)[0, 1]
        # Closed cases are a selected, fraud-heavy sample; shrink logits toward
        # the neutral benchmark prior rather than exposing overconfident scores.
        probability = 0.5 + 0.5 * (raw_probability - 0.5)
        return float(min(0.99, max(0.01, probability)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> TabularFraudModel:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a TabularFraudModel")
        return model


def train_from_duckdb(database_path: Path) -> tuple[TabularFraudModel, int]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        frame = connection.execute(
            """
            WITH labels AS (
              SELECT outcome, unnest(string_split(txn_ids, '|')) AS txn_id
              FROM closed_cases
              WHERE txn_ids IS NOT NULL AND txn_ids <> ''
            )
            SELECT t.*, i.* EXCLUDE ("TransactionID"), labels.outcome
            FROM labels
            JOIN transactions t ON t."TransactionID" = labels.txn_id
            LEFT JOIN identity i USING ("TransactionID")
            """
        ).fetchdf()
    labels = [1 if value == "confirmed_fraud" else 0 for value in frame.pop("outcome")]
    rows = frame.to_dict(orient="records")
    return TabularFraudModel().fit_rows(rows, labels), len(rows)
