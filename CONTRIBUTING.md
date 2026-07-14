# Contributing

Keep Tilia's public contracts coherent and dependencies minimal. New estimators
must declare capabilities and include conformance, mathematical-invariant, and
regression tests as applicable. Core behavior must be implemented in this
repository rather than delegated to another machine-learning framework.

Run the test suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Reactant, DifferentiationInterface, and Makie tests use the persistent
repository environments under `test/`; do not add them as hard dependencies.
Keep Reactant precompilation serial with `JULIA_NUM_PRECOMPILE_TASKS=1`.

Benchmarks run from the persistent `benchmark` environment. Report compilation
or first-call latency separately from steady-state timing, and include small,
medium, and large inputs where relevant.
