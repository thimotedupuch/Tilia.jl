# Small Tilia–scikit-learn performance snapshot

Measured 2026-07-15. Lower is better.

| Task | Shape/configuration | Tilia | sklearn | Tilia / sklearn |
|---|---:|---:|---:|---:|
| Linear regression fit | 20,000 × 32 | 9.8 ms | 15.6 ms | 0.63× |
| PCA fit | 10,000 × 32, 8 components | 1.5 ms | 1.5 ms | 1.00× |
| K-means fit | 5,000 × 16, 8 clusters, 3 starts | 41.9 ms | 27.8 ms | 1.51× |
| Brute-force neighbor query | 200 against 10,000 × 16, k=5 | 14.9 ms | 7.5 ms | 1.99× |
| Decision-tree fit | 1,000 × 12, depth ≤ 6 | 4.8 ms | 5.2 ms | 0.92× |

Dense regression remains faster, while PCA and the binary tree are close.
K-means dropped from 83.2 ms to 38.5–45.4 ms after removing duplicate distance
matrices; brute-force neighbors remain the clearest roughly 2× gap.

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
