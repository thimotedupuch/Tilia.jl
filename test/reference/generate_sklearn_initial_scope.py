"""Generate cross-family sklearn reference values for Tilia's initial scope."""

from pathlib import Path
import json

import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


X = np.array([
    [-3.0, -1.0, 0.0], [-2.5, -0.5, 1.0], [-2.0, -1.5, 0.5],
    [-1.0, -0.5, 1.5], [-0.5, -1.0, 2.0], [0.5, 1.0, -1.0],
    [1.0, 0.5, -1.5], [2.0, 1.5, -0.5], [2.5, 0.5, -1.0],
    [3.0, 1.0, -2.0],
])
y_reg = X @ np.array([1.5, -0.75, 0.4]) + 0.3
y_class = np.where(X[:, 0] + 0.25 * X[:, 1] > 0, "positive", "negative")
query = np.array([[-2.25, -0.75, 0.5], [0.0, 0.0, 0.0], [2.25, 1.0, -1.0]])

linear = LinearRegression().fit(X, y_reg)
ridge = Ridge(alpha=1.0).fit(X, y_reg)
lasso = Lasso(alpha=0.05, max_iter=10_000, tol=1e-10).fit(X, y_reg)
logistic = LogisticRegression(C=1.0, solver="lbfgs", random_state=0).fit(X, y_class)
neighbors = NearestNeighbors(n_neighbors=3, algorithm="brute").fit(X)
neighbor_distances, neighbor_indices = neighbors.kneighbors(query)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y_class)
kernel_ridge = KernelRidge(alpha=0.2, kernel="rbf", gamma=0.5).fit(X, y_reg)
svc = SVC(C=2.0, kernel="linear", random_state=0).fit(X, y_class)


def encoded(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, separators=(", ", ": "))


sections = {
    "source": {
        "package": "scikit-learn",
        "version": sklearn.__version__,
        "generator": "test/reference/generate_sklearn_initial_scope.py",
    },
    "tolerance": {"absolute": 1e-4, "relative": 1e-4},
    "input": {"X": X, "y_reg": y_reg, "y_class": y_class, "query": query},
    "linear_regression": {"coefficients": linear.coef_, "intercept": linear.intercept_,
                          "prediction": linear.predict(query)},
    "ridge": {"coefficients": ridge.coef_, "intercept": ridge.intercept_,
              "prediction": ridge.predict(query)},
    "lasso": {"coefficients": lasso.coef_, "intercept": lasso.intercept_,
              "prediction": lasso.predict(query)},
    "logistic": {"prediction": logistic.predict(query)},
    "neighbors": {"distances": neighbor_distances, "indices_zero_based": neighbor_indices},
    "decision_tree": {"prediction": tree.predict(query)},
    "kernel_ridge": {"prediction": kernel_ridge.predict(query)},
    "support_vector": {"prediction": svc.predict(query)},
}

lines = []
for section, values in sections.items():
    lines.append(f"[{section}]")
    for key, value in values.items():
        lines.append(f"{key} = {encoded(value)}")
    lines.append("")

Path(__file__).with_name("initial_scope_sklearn.toml").write_text(
    "\n".join(lines), encoding="utf-8"
)
