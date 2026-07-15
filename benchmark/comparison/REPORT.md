# Small Tilia–scikit-learn performance snapshot

Measured 2026-07-15. Lower is better.

| Task | Shape/configuration | Tilia | sklearn | Tilia / sklearn |
|---|---:|---:|---:|---:|
| Linear regression fit | 20,000 × 32 | 12.3 ms | 19.2 ms | 0.64× |
| PCA fit | 10,000 × 32, 8 components | 10.1 ms | 2.2 ms | 4.65× |
| K-means fit | 5,000 × 16, 8 clusters, 3 starts | 427.7 ms | 44.6 ms | 9.58× |
| Brute-force neighbor query | 200 against 10,000 × 16, k=5 | 33.0 ms | 12.1 ms | 2.73× |
| Decision-tree fit | 1,000 × 12, depth ≤ 6 | 86.9 ms | 7.8 ms | 11.09× |

This confirms the current expectation: dense factorization-based regression is
competitive, while PCA trails sklearn and allocation-heavy k-means, neighbors,
and tree paths need further optimization.

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
