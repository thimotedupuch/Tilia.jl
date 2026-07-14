"""Regenerate phase-B offline reference values with scikit-learn."""

import json
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

X = np.array([
    [-3.2, -2.9], [-3.0, -3.1], [-2.8, -3.0], [-3.1, -2.8],
    [2.8, 3.1], [3.0, 2.9], [3.2, 3.0], [3.1, 3.2],
])
y = np.array(["cold"] * 4 + ["hot"] * 4)
query = np.array([[-3.0, -3.0], [3.0, 3.0], [0.0, 0.0]])

pca = PCA(n_components=2, svd_solver="full").fit(X)
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1,
                max_iter=100, tol=1e-5, random_state=0).fit(X)
gnb = GaussianNB().fit(X, y)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1e-6).fit(X, y)
qda = QuadraticDiscriminantAnalysis(reg_param=1e-6).fit(X, y)
mixture = GaussianMixture(n_components=2, covariance_type="full", n_init=1,
                          max_iter=100, tol=1e-5, reg_covar=1e-5,
                          random_state=0).fit(X)
neighbors = KNeighborsRegressor(n_neighbors=2, weights="distance").fit(
    X, np.arange(8, dtype=float)
)

ordered_kmeans = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 0])]
order = np.argsort(mixture.means_[:, 0])
result = {
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "input": {"X": X.tolist(), "y": y.tolist(), "query": query.tolist()},
    "pca": {
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    },
    "kmeans": {"centers": ordered_kmeans.tolist(), "inertia": kmeans.inertia_},
    "gaussian_nb": {"probabilities": gnb.predict_proba(query).tolist()},
    "lda": {"probabilities": lda.predict_proba(query).tolist()},
    "qda": {"probabilities": qda.predict_proba(query).tolist()},
    "gaussian_mixture": {
        "means": mixture.means_[order].tolist(),
        "weights": mixture.weights_[order].tolist(),
    },
    "neighbors": {"prediction": neighbors.predict(query).tolist()},
    "tolerance": {"absolute": 1e-5, "relative": 1e-5},
}
print(json.dumps(result, indent=2, sort_keys=True))
