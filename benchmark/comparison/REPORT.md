# Small Tilia–scikit-learn performance snapshot

Measured 2026-07-15. Lower is better.

| Task | Shape/configuration | Tilia | sklearn | Tilia / sklearn |
|---|---:|---:|---:|---:|
| Linear regression fit | 20,000 × 32 | 14.5 ms | 19.2 ms | 0.76× |
| PCA fit | 10,000 × 32, 8 components | 1.7 ms | 2.2 ms | 0.75× |
| K-means fit | 5,000 × 16, 8 clusters, 3 starts | 88.6 ms | 44.6 ms | 1.99× |
| Brute-force neighbor query | 200 against 10,000 × 16, k=5 | 15.8 ms | 12.1 ms | 1.31× |
| Decision-tree fit | 1,000 × 12, depth ≤ 6 | 4.6 ms | 7.8 ms | 0.60× |

This confirms the current expectation: dense factorization-based regression is
competitive on the dense regression, PCA, and binary-tree cases. K-means and
brute-force neighbors are within roughly 2×.

Setup: Julia 1.12.4, Tilia 0.1.0-DEV, scikit-learn 1.9.0, x86-64 Linux,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Values are medians of three
measured warm runs with garbage collection before each run. Inputs have matching
shapes and distributions but are not bit-identical. Default estimator workflows
were used, so solver choices can differ. This is a development snapshot, not a
comprehensive benchmark.

Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=benchmark benchmark/comparison/run_tilia.jl
cd sklearn_tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run python benchmark_comparison.py
```
