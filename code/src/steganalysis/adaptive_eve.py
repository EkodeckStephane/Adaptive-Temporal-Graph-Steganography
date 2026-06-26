from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

PUBLIC_FEATURE_COLUMNS = [
    "action_probability",
    "surprise_bits",
    "rank_fraction",
    "is_top_action",
    "entropy_bits",
    "top_probability",
    "candidate_count",
    "unseen_context",
    "unseen_destination",
    "same_as_previous",
    "self_loop",
    "log_inter_event_gap",
]

BOUNDED_WHITE_BOX_FEATURE_COLUMNS = [
    *PUBLIC_FEATURE_COLUMNS,
    "encoder_capacity_bits",
    "encoder_tv_bound",
    "encoder_kl_bound",
    "embedding_feasible",
]

ORACLE_LEAKAGE_FEATURE_COLUMNS = [
    *BOUNDED_WHITE_BOX_FEATURE_COLUMNS,
    # These variables are not observable without the secret bitstream or an
    # instrumentation leak. They are retained as an impossibility/audit test.
    "bits_consumed",
    "local_total_variation",
    "local_kl_bits",
]


@dataclass(frozen=True)
class AdaptiveEveMetrics:
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    balanced_accuracy: float
    eer: float
    feature_importance: dict[str, float]


def fit_and_score_adaptive_eve(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    feature_columns: list[str] | tuple[str, ...] = BOUNDED_WHITE_BOX_FEATURE_COLUMNS,
    n_estimators: int = 200,
    max_depth: int = 5,
    bootstrap_rounds: int = 200,
) -> AdaptiveEveMetrics:
    """Train an adaptive steganalyst under an explicit feature budget.

    The default is a bounded white-box adversary: it knows the cover model,
    codec, and embedding rule, but not the key, plaintext, or secret bitstream.
    Oracle-leakage features must be requested explicitly.
    """
    available = [column for column in feature_columns if column in validation.columns]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(validation[available].to_numpy(dtype=float))
    train_y = validation["label"].to_numpy(dtype=int)
    test_x = scaler.transform(test[available].to_numpy(dtype=float))
    test_y = test["label"].to_numpy(dtype=int)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]
    predictions = model.predict(test_x)
    auc = float(roc_auc_score(test_y, scores))
    ci_low, ci_high = _bootstrap_auc_ci(test_y, scores, seed=seed, rounds=bootstrap_rounds)
    return AdaptiveEveMetrics(
        auc=auc,
        auc_ci_low=ci_low,
        auc_ci_high=ci_high,
        balanced_accuracy=float(balanced_accuracy_score(test_y, predictions)),
        eer=_eer(test_y, scores),
        feature_importance={
            column: float(importance)
            for column, importance in zip(available, model.feature_importances_, strict=True)
        },
    )


def metrics_to_dict(metrics: AdaptiveEveMetrics) -> dict[str, float]:
    return {
        "auc": metrics.auc,
        "auc_ci_low": metrics.auc_ci_low,
        "auc_ci_high": metrics.auc_ci_high,
        "balanced_accuracy": metrics.balanced_accuracy,
        "eer": metrics.eer,
    }


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    return float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)


def _bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    rounds: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    values = []
    for _ in range(rounds):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(roc_auc_score(sampled_labels, scores[indices]))
    if not values:
        auc = float(roc_auc_score(labels, scores))
        return auc, auc
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
