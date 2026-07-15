# Reactant CPU snapshot

Measured 2026-07-15 with Julia 1.12.4, Reactant on CPU, and scikit-learn
1.9.0. Times are milliseconds; lower is better.

| Rows × features | Vanilla fit | Reactant fit | sklearn fit |
|---:|---:|---:|---:|
| 100 × 16 | 7,447.4* | 29,292.8* | 17.1 |
| 1,000 × 16 | 37.4 | 900.5 | 4.8 |
| 10,000 × 16 | 52.8 | 864.0 | 11.4 |

| Rows × features | Vanilla warm inference | Reactant warm inference | sklearn warm inference |
|---:|---:|---:|---:|
| 100 × 16 | 0.016 | 0.043 | 0.656 |
| 1,000 × 16 | 0.048 | 0.102 | 0.725 |
| 10,000 × 16 | 0.341 | 0.745 | 1.192 |

Reactant compilation was 0.82–0.86 seconds per new shape after initial package
compilation. On this CPU workload, Reactant warm inference beats sklearn but
does not beat vanilla Tilia; compilation makes it unsuitable for these small
one-off fits. `*` includes initial Julia/extension compilation and is not a
steady-state fit comparison.

Reproduce with:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/accelerator \
  benchmark/accelerator/runbenchmarks.jl
cd sklearn_tests
pixi run python benchmark_reactant_cpu.py
```
