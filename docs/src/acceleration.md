# Acceleration

Reactant is a weak dependency and lives in the persistent repository test
environment `test/accelerator`. Core Tilia does not install or load it.

```julia
using Tilia, Reactant
context = FitContext(backend=ReactantBackend(device=:cpu))
fitted = fit(Chain(Standardize(), LogisticRegression()), X, y; context)
```

Reports distinguish compilation, transfer, device execution, unsupported
operations, and explicit CPU fallback. Accelerator precompilation should use
`JULIA_NUM_PRECOMPILE_TASKS=1`.
