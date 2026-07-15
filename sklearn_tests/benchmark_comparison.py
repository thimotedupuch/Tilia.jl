import gc
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

REPEATS = 3


def median_ms(operation):
    operation()
    timings = []
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - start)
    return np.median(timings) * 1_000


print("task\tmedian_ms")

rng = np.random.default_rng(20260715)
X = rng.standard_normal((20_000, 32))
y = X @ rng.standard_normal(32) + 0.1 * rng.standard_normal(20_000)
print(f"linear_regression_fit\t{median_ms(lambda: LinearRegression().fit(X, y)):.3f}")

rng = np.random.default_rng(20260715)
X = rng.standard_normal((10_000, 32))
print(f"pca_fit\t{median_ms(lambda: PCA(n_components=8).fit(X)):.3f}")

rng = np.random.default_rng(20260715)
X = rng.standard_normal((5_000, 16))
print(f"kmeans_fit\t{median_ms(lambda: KMeans(n_clusters=8, n_init=3,
      max_iter=50, algorithm='lloyd', random_state=20260715).fit(X)):.3f}")

rng = np.random.default_rng(20260715)
X = rng.standard_normal((10_000, 16))
query = rng.standard_normal((200, 16))
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(X)
print(f"neighbors_query\t{median_ms(lambda: neighbors.kneighbors(query)):.3f}")

rng = np.random.default_rng(20260715)
X = rng.standard_normal((1_000, 12))
y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "positive", "negative")
print(f"decision_tree_fit\t{median_ms(lambda: DecisionTreeClassifier(
      max_depth=6, random_state=20260715).fit(X, y)):.3f}")
