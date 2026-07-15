"""Regenerate the offline DBSCAN reference values."""

import json
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN

X = np.array([[0., 0.], [.1, 0.], [0., .1], [3., 3.],
              [3.1, 3.], [3., 3.1], [8., 8.]])
model = DBSCAN(eps=.25, min_samples=3).fit(X)
print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "case": {"X": X.tolist(), "core_indices": model.core_sample_indices_.tolist(),
             "cluster_labels": model.labels_.tolist()},
}, indent=2, sort_keys=True))
