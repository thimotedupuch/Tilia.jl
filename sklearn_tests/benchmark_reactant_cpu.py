"""Small CPU baseline corresponding to benchmark/accelerator/runbenchmarks.jl."""

import gc
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def elapsed(operation):
    gc.collect()
    started = time.perf_counter()
    result = operation()
    return time.perf_counter() - started, result


print(f"scikit-learn={sklearn.__version__}")
for observations in (100, 1_000, 10_000):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((observations, 16), dtype=np.float32)
    y = np.where(np.arange(observations) < observations // 2,
                 "negative", "positive")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=100, random_state=42),
    )
    fit_seconds, _ = elapsed(lambda: model.fit(X, y))
    first_seconds, _ = elapsed(lambda: model.predict_proba(X))
    samples = [elapsed(lambda: model.predict_proba(X))[0] for _ in range(3)]
    print({
        "benchmark": "sklearn_cpu_scaling",
        "observations": observations,
        "features": X.shape[1],
        "fit_wall_seconds": fit_seconds,
        "first_call_seconds": first_seconds,
        "steady_state_seconds": float(np.median(samples)),
    })
