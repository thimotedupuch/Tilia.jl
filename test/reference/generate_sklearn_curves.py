"""Regenerate binary curve and calibration references with scikit-learn."""

import json

import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve


targets = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, roc_thresholds = roc_curve(targets, scores, drop_intermediate=False)
precision, recall, pr_thresholds = precision_recall_curve(targets, scores)
fraction_positive, mean_predicted = calibration_curve(
    targets, scores, n_bins=2, strategy="uniform"
)

print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "input": {"targets": targets.tolist(), "scores": scores.tolist()},
    "roc": {
        "false_positive_rate": fpr.tolist(),
        "true_positive_rate": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
        "area": auc(fpr, tpr),
    },
    "precision_recall": {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    },
    "calibration": {
        "fraction_positive": fraction_positive.tolist(),
        "mean_predicted_probability": mean_predicted.tolist(),
    },
    "tolerance": {"absolute": 1e-12, "relative": 1e-12},
}, indent=2, sort_keys=True))
