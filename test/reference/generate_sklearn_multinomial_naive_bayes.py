"""Regenerate the offline multinomial naive Bayes reference values."""

import json
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB

X = np.array([[3., 0, 1], [2, 0, 0], [0, 3, 1],
              [0, 2, 2], [4, 0, 0], [0, 4, 1]])
y = np.array(["left", "left", "right", "right", "left", "right"])
model = MultinomialNB(alpha=1.0).fit(X, y)
print(json.dumps({
    "source": {"package": "scikit-learn", "version": sklearn.__version__},
    "case": {"X": X.tolist(), "y": y.tolist(),
             "classes": model.classes_.tolist(),
             "feature_log_probabilities": model.feature_log_prob_.tolist(),
             "probabilities": model.predict_proba(X).tolist()},
    "tolerance": {"absolute": 1e-12, "relative": 1e-12},
}, indent=2, sort_keys=True))
