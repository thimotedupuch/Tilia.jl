"""Regenerate the offline feature-agglomeration reference."""

import json
import numpy as np
import sklearn
from sklearn.cluster import FeatureAgglomeration

x = np.linspace(-2.0, 2.0, 5)
z = np.sin(3 * x)
X = np.column_stack((x, x + .001 * np.cos(x), z, z + .001 * x))
model = FeatureAgglomeration(n_clusters=2, linkage="average").fit(X)
print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "case": {"X": X.tolist(), "labels": model.labels_.tolist()},
}, indent=2, sort_keys=True))
