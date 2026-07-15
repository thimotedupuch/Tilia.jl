"""Regenerate offline references for common numeric feature transforms."""

import json

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler


X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 2.0]])
query = np.array([[0.0, 3.0], [6.0, 5.0]])

minimum = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X)
normalizer = Normalizer(norm="l2").fit(X)
robust = RobustScaler(quantile_range=(25.0, 75.0)).fit(X)
polynomial = PolynomialFeatures(degree=2, include_bias=True).fit(X)

print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "input": {"X": X.tolist(), "query": query.tolist()},
    "minmax": {
        "data_min": minimum.data_min_.tolist(),
        "data_range": minimum.data_range_.tolist(),
        "query": minimum.transform(query).tolist(),
    },
    "normalize": {"query": normalizer.transform(query).tolist()},
    "robust": {
        "center": robust.center_.tolist(),
        "scale": robust.scale_.tolist(),
        "query": robust.transform(query).tolist(),
    },
    "polynomial": {
        "powers": polynomial.powers_.tolist(),
        "training": polynomial.transform(X).tolist(),
    },
    "tolerance": {"absolute": 1e-12, "relative": 1e-12},
}, indent=2, sort_keys=True))
