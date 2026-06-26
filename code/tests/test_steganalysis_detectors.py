from __future__ import annotations

import numpy as np

from steganalysis.detectors import fit_detector, score_detector


def test_detector_metrics_are_bounded() -> None:
    x_train = np.asarray([[0.0], [0.1], [1.0], [1.1]])
    y_train = np.asarray([0, 0, 1, 1])
    detector = fit_detector("linear", x_train, y_train, seed=3)

    metrics = score_detector(detector, x_train, y_train)

    assert 0 <= metrics.auc <= 1
    assert 0 <= metrics.balanced_accuracy <= 1
    assert 0 <= metrics.eer <= 1
