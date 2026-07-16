# Numerical behavior

Numerical choices in Tilia are explicit configuration and reportable state.
`FitContext` owns backend choice, random streams, numerical policy,
determinism, and compilation cache; estimators derive operation-specific
contexts rather than consuming Julia's global RNG.

## Fit context

```julia
context = FitContext(
    backend=CPUBackend(),
    seed=42,
    numerics=NumericsPolicy(Float64),
    deterministic=true,
)

fitted = fit(model, X, y; context=context)
```

`root_seed` and `stream_id` appear in fit reports. `derive_context` creates a
named deterministic substream while sharing immutable policy and compilation
cache:

```julia
fold_context = derive_context(context, :cross_validation, :fold, 1)
```

This makes stochastic meaning independent of unrelated global RNG draws and,
for named tasks, scheduling order.

## Numerical policy

`NumericsPolicy` controls shared defaults:

| Setting | Purpose |
|:--|:--|
| `float_type`, `accumulation_type` | Working and reduction precision |
| `tolerance`, `tolerance_scale` | Base and scale-aware convergence tolerance |
| `max_iterations` | Context-wide upper bound for iterative work |
| `stable_summation` | Stable accumulation where implemented |
| `missing_policy`, `finite_policy` | Handling contract for exceptional inputs |
| `overflow_policy`, `underflow_policy` | Exceptional floating-point behavior |
| `deterministic_reductions` | Reproducible reduction requirement |
| `sparse_centering` | Error or explicit densification policy |

Constructor-specific iteration limits are combined with the context maximum;
the context can tighten an estimator request but does not silently relax it.

```julia
policy = NumericsPolicy(
    Float32;
    accumulation_type=Float64,
    max_iterations=500,
    sparse_centering=:error,
)
```

## Centralized kernels and solvers

Stable sigmoid, log-sum-exp, log-softmax, binary loss, norms, weighted
statistics, covariance, distances, ranking, sparse operations, and regression
metrics live in `Tilia.Kernels`. Factorization and optimization routines live
in `Tilia.Solvers`. Statistical model files call these shared implementations
instead of embedding slightly different primitives.

Float32 and Float64 behavior is exercised across the conformance matrix.
Models preserve supported floating input types for numeric predictions and
transforms; individual objectives and reports document any promoted
accumulation.

## Convergence and failure

Iterative fitted objects and reports expose relevant fields such as objective
history, iteration count, convergence flag, tolerance, and warnings. Numerical
problems raise typed Tilia errors rather than being converted to plausible
predictions or silent `NaN` values.

Typical explicit failures include:

- non-finite numeric input;
- rank or covariance problems outside the configured policy;
- unsupported sparse centering;
- schema or feature-count mismatch;
- invalid probabilities, weights, or hyperparameters;
- unavailable backend operations without an allowed fallback.

Exact objective, regularization, stopping, and intercept conventions are listed
in [Model numerical contracts](model-semantics.md). Backend behavior is covered
by [Acceleration](acceleration.md).
