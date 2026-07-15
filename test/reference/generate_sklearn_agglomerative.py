"""Regenerate the offline agglomerative clustering reference."""

import json
import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering

X = np.array([[0., 0.], [.1, 0.], [0., .1], [4., 4.], [4.1, 4.], [4., 4.1]])
model = AgglomerativeClustering(n_clusters=2, linkage="average").fit(X)
print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "case": {"X": X.tolist(), "labels": model.labels_.tolist()},
}, indent=2, sort_keys=True))
