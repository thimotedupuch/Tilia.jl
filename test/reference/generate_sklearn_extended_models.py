"""Regenerate offline references for Tilia's extended supervised models.

Run from the repository root with:

    pixi run --manifest-path sklearn_tests/pixi.toml \
        python test/reference/generate_sklearn_extended_models.py

The ordinal and calibration optimizations deliberately use SciPy finite-
difference gradients, independently of Tilia's analytic-gradient solvers.
Standard output is the complete TOML fixture.
"""

import json

import numpy as np
import scipy
from scipy.optimize import minimize
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import (
    GammaRegressor,
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    QuantileRegressor,
    RANSACRegressor,
    Ridge,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.model_selection import KFold, cross_val_predict


X_REG = np.array([
    [-2.0, 0.5], [-1.5, 1.2], [-1.0, -0.4], [-0.5, 1.8],
    [0.0, 0.3], [0.5, -1.0], [1.0, 1.1], [1.5, -0.6],
    [2.0, 0.8], [2.5, -1.4], [3.0, 1.5], [3.5, -0.2],
])
Y_POSITIVE = np.array([0.8, 1.4, 1.1, 2.2, 2.0, 1.7, 3.5, 2.8, 4.6, 3.7, 6.2, 5.4])
Y_ROBUST = np.array([-2.7, -1.8, -1.9, -0.2, 0.1, 0.0, 2.2, 2.1, 4.0, 4.1, 12.0, 5.8])
QUERY_REG = np.array([[-1.25, 0.2], [0.75, 0.7], [2.75, -0.3]])

X_ORDINAL = np.array([
    [-2.4, 0.2], [-2.0, -0.5], [-1.7, 0.8], [-1.3, -0.2], [-0.9, 0.4],
    [-0.5, -0.7], [-0.1, 0.6], [0.2, -0.1], [0.5, 0.9], [0.9, -0.5],
    [1.2, 0.4], [1.5, -0.8], [1.9, 0.7], [2.2, -0.3], [2.6, 0.5],
])
Y_ORDINAL = np.array([0, 0, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2])
QUERY_ORDINAL = np.array([[-1.5, 0.0], [0.0, 0.3], [1.8, -0.2]])

X_BINARY = np.array([
    [-2.4, 0.1], [-1.8, -0.6], [-1.2, 0.8], [-0.7, -0.2],
    [0.2, 0.6], [0.7, -0.8], [1.2, 0.4], [1.8, -0.3],
    [-1.5, 0.5], [-0.2, -0.4], [0.9, 0.9], [2.3, -0.7],
])
Y_BINARY = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1])
QUERY_BINARY = np.array([[-1.0, 0.0], [0.0, 0.2], [1.5, -0.1]])


def rows(values):
    return np.asarray(values).tolist()


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return repr(value)


def dump_toml(document):
    blocks = []
    for section, values in document.items():
        blocks.append(f"[{section}]")
        blocks.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        blocks.append("")
    return "\n".join(blocks)


def ordinal_probabilities(parameters, X, class_count):
    feature_count = X.shape[1]
    beta = parameters[:feature_count]
    alpha = parameters[feature_count:]
    thresholds = np.empty(class_count - 1)
    thresholds[0] = alpha[0]
    for index in range(1, class_count - 1):
        thresholds[index] = thresholds[index - 1] + np.exp(alpha[index])
    eta = X @ beta
    probabilities = np.empty((len(X), class_count))
    for row, value in enumerate(eta):
        cumulative = 1.0 / (1.0 + np.exp(-(thresholds - value)))
        probabilities[row, 0] = cumulative[0]
        probabilities[row, -1] = 1.0 - cumulative[-1]
        for level in range(1, class_count - 1):
            probabilities[row, level] = cumulative[level] - cumulative[level - 1]
    return probabilities


def ordinal_objective(parameters):
    probabilities = ordinal_probabilities(parameters, X_ORDINAL, 3)
    selected = probabilities[np.arange(len(Y_ORDINAL)), Y_ORDINAL]
    return -np.log(np.maximum(selected, 1e-15)).sum()


ordinal_initial = np.zeros(X_ORDINAL.shape[1] + 2)
ordinal_initial[X_ORDINAL.shape[1]] = -1.0
ordinal = minimize(ordinal_objective, ordinal_initial, method="L-BFGS-B",
                   options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000})

cv = KFold(n_splits=3, shuffle=False)
base_classifier = LogisticRegression(C=1e12, solver="lbfgs", max_iter=2000, tol=1e-10)
oof_probability = cross_val_predict(
    base_classifier, X_BINARY, Y_BINARY, cv=cv, method="predict_proba"
)[:, 1]


def platt_objective(parameters):
    logits = parameters[0] * oof_probability + parameters[1]
    probability = 1.0 / (1.0 + np.exp(-logits))
    probability = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return -(Y_BINARY * np.log(probability) +
             (1 - Y_BINARY) * np.log(1 - probability)).sum()


platt = minimize(platt_objective, np.array([1.0, 0.0]), method="BFGS",
                 options={"gtol": 1e-10, "maxiter": 2000})
final_classifier = base_classifier.fit(X_BINARY, Y_BINARY)
query_base_probability = final_classifier.predict_proba(QUERY_BINARY)[:, 1]
query_calibrated_probability = 1.0 / (
    1.0 + np.exp(-(platt.x[0] * query_base_probability + platt.x[1]))
)

stacking = StackingRegressor(
    estimators=[
        ("linear", LinearRegression()),
        ("ridge", Ridge(alpha=0.5, solver="cholesky")),
    ],
    final_estimator=Ridge(alpha=0.25, solver="cholesky"),
    cv=cv,
).fit(X_REG, Y_ROBUST)

models = {
    "poisson": PoissonRegressor(alpha=0.0, max_iter=2000, tol=1e-10),
    "gamma": GammaRegressor(alpha=0.0, max_iter=2000, tol=1e-10),
    "tweedie": TweedieRegressor(power=1.5, alpha=0.0, link="log", max_iter=2000, tol=1e-10),
}
glm_predictions = {
    name: model.fit(X_REG, Y_POSITIVE).predict(QUERY_REG).tolist()
    for name, model in models.items()
}

robust_predictions = {
    "quantile": QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
        .fit(X_REG, Y_ROBUST).predict(QUERY_REG).tolist(),
    "huber": HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=2000, tol=1e-10)
        .fit(X_REG, Y_ROBUST).predict(QUERY_REG).tolist(),
    "theil_sen": TheilSenRegressor(max_subpopulation=10000, random_state=0)
        .fit(X_REG, Y_ROBUST).predict(QUERY_REG).tolist(),
    "ransac": RANSACRegressor(
        estimator=LinearRegression(), min_samples=3, residual_threshold=0.5,
        max_trials=100, stop_probability=0.99, loss="absolute_error",
        random_state=0,
    ).fit(X_REG, Y_ROBUST).predict(QUERY_REG).tolist(),
}

result = {
    "source": {
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "generator": "test/reference/generate_sklearn_extended_models.py",
    },
    "tolerance": {
        "glm_absolute": 1e-4,
        "glm_relative": 1e-4,
        "robust_absolute": 1e-1,
        "robust_relative": 2.5e-2,
        "ordinal_absolute": 1e-4,
        "ordinal_relative": 1e-4,
        "calibration_absolute": 2e-3,
        "calibration_relative": 2e-3,
        "stacking_absolute": 1e-6,
        "stacking_relative": 1e-6,
    },
    "input": {
        "X_regression": rows(X_REG),
        "y_positive": rows(Y_POSITIVE),
        "y_robust": rows(Y_ROBUST),
        "query_regression": rows(QUERY_REG),
        "X_ordinal": rows(X_ORDINAL),
        "y_ordinal": rows(Y_ORDINAL),
        "query_ordinal": rows(QUERY_ORDINAL),
        "X_binary": rows(X_BINARY),
        "y_binary": rows(Y_BINARY),
        "query_binary": rows(QUERY_BINARY),
    },
    "glm": glm_predictions,
    "robust": robust_predictions,
    "ordinal": {
        "optimizer_success": bool(ordinal.success),
        "objective": float(ordinal.fun),
        "probabilities": rows(ordinal_probabilities(ordinal.x, QUERY_ORDINAL, 3)),
    },
    "calibration": {
        "platt_parameters": rows(platt.x),
        "probabilities": rows(np.column_stack(
            [1.0 - query_calibrated_probability, query_calibrated_probability]
        )),
    },
    "stacking_regression": {
        "prediction": rows(stacking.predict(QUERY_REG)),
    },
}

print(dump_toml(result), end="")
