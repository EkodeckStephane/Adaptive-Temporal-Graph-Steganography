from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DetectorMetrics:
    auc: float
    balanced_accuracy: float
    eer: float


def make_detector(name: str, *, seed: int) -> object:
    if name == "linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        alpha=1e-3,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown detector: {name}")


def fit_detector(name: str, x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> object:
    detector = make_detector(name, seed=seed)
    detector.fit(x_train, y_train)
    return detector


def score_detector(detector: object, x_test: np.ndarray, y_test: np.ndarray) -> DetectorMetrics:
    scores = detector.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return DetectorMetrics(
        auc=float(roc_auc_score(y_test, scores)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        eer=_eer(y_test, scores),
    )


def metrics_to_dict(metrics: DetectorMetrics) -> dict[str, float]:
    return {
        "auc": metrics.auc,
        "balanced_accuracy": metrics.balanced_accuracy,
        "eer": metrics.eer,
    }


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    return float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)
