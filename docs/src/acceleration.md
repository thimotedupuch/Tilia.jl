# Acceleration

Reactant integration is optional. Core Tilia neither installs nor loads
Reactant, and CPU execution remains the authoritative implementation. The
extension activates only when both packages are present in the environment.

## Current supported path

The present prototype accelerates dense inference for the semantic chain
`Standardize() → LogisticRegression()` and compiles a logistic objective.
Training statistics and the Newton solver still execute explicitly on CPU.
This is reported as a mixed execution path, not described as full-device
training.

```julia
using Tilia
using Reactant

context = FitContext(
    backend=ReactantBackend(device=:cpu),
    seed=42,
)

model = Chain(Standardize(), LogisticRegression())
fitted = fit(model, X, y; context=context)
probabilities = predict_proba(fitted, Xnew)
```

`device` may be `:auto`, `:cpu`, or `:gpu`, subject to the locally installed
Reactant/XLA platform.

## Unsupported graphs and fallback

The default fallback policy is `:error`:

```julia
ReactantBackend(fallback=:error)
```

An unsupported graph or compilation failure raises `UnsupportedBackendError`.
To opt into an explicit host fallback:

```julia
context = FitContext(
    backend=ReactantBackend(device=:auto, fallback=:cpu),
)
```

Fallback returns a CPU fitted graph with warnings and structured details. It is
never silently presented as accelerated execution.

## Reports and compilation cache

```julia
details = report(fitted).details
```

Accelerator reports distinguish:

- requested and actual backend and device;
- accelerated, host, and host-fit nodes;
- input/output transfer locations and byte counts;
- compilation and last-execution time;
- compilation-cache hits;
- unsupported and fallback operations;
- whether the logistic objective was accelerated.

A `CompilationCache` belongs to `FitContext` and is shared by contexts derived
from it. Compatible input signatures may reuse compiled entries; reports expose
the observed hits rather than requiring users to infer them from timing.

## Persistence behavior

Saving a fitted Reactant graph stores its authoritative CPU fitted graph. This
keeps model artifacts independent of a particular device runtime. Reloading
does not claim to restore compiled executables or accelerator cache state.

## Development environment

Accelerator dependencies and tests live in their own persistent environment:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/accelerator \
    -e 'using Pkg; Pkg.instantiate()'

JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/accelerator \
    test/accelerator/runtests.jl
```

Keep `JULIA_NUM_PRECOMPILE_TASKS=1` for this environment. See
[Pipelines and graphs](pipelines-and-graphs.md) for backend-neutral graph
placement and [Numerical behavior](numerical-behavior.md) for context policy.
