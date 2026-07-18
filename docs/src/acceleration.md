# Acceleration

Reactant integration is optional. Core Tilia neither installs nor loads
Reactant, and CPU execution remains the authoritative implementation. The
extension activates only when both packages are present in the environment.

## Current supported path

The extension compiles dense Float32/Float64 inference regions composed from
standardization, min--max scaling and clipping, numeric imputation, selection,
concatenation, PCA/truncated-SVD projection, and linear, ridge, or logistic
prediction heads. Logistic class selection runs on device. Eligible linear
graphs beginning with `Standardize` compute population statistics on Reactant
in the configured accumulation type and refit downstream nodes against those
statistics. Other transforms and model solvers remain explicit CPU work; the
fitted CPU graph remains the portable model state.

Weighted ridge fitting with positive regularization and `solver=:cholesky`
keeps centering, weighted Gram and cross-product construction, the regularized
Cholesky solve, intercept recovery, and residual-norm evaluation on Reactant.
Only fitted coefficients and scalar diagnostics return to the host. QR and SVD
models keep their original CPU algorithms.

Supported whole-graph logistic and ridge fits construct portable placeholder
nodes and replace them with Reactant-fitted parameters, avoiding duplicate CPU
transform and solver fits.

Logistic fitting uses a device-resident damped Newton loop for supported linear
graphs. Weighted gradients, Hessians, Cholesky solves, convergence decisions,
and batched Armijo candidates stay on Reactant. Reports materialize the complete
bounded objective history once per one-vs-rest class after the device loop.

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

Backend selection is process-global in the current Reactant API. Tilia
serializes its Reactant fits and predictions, restores the previous backend in
a `finally` block, locks shared compilation caches, and serializes execution of
each fitted object. Concurrent Tilia fits and predictions are therefore safe;
external code that directly changes Reactant's default backend is outside this
lock and must coordinate separately.

Existing `Reactant.AbstractConcreteArray` matrices are accepted without
re-uploading the input. Probabilistic and regression inference can keep results
on the device:

```julia
device_X = Reactant.to_rarray(Xnew)
device_probabilities = predict_proba(fitted, device_X; output=:device)
```

`output` is `:host` by default and may be `:device`. Classification `predict`
always returns host labels because labels may be arbitrary Julia values; use
`predict_proba(...; output=:device)` when downstream work stays on-device.

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

With `fallback=:cpu`, supported and unsupported inference nodes are partitioned
into regions for linear chains and branched DAGs. Reports and the numerical
execution graph show host/device placement and transfers on the actual graph
edges. Clipped min--max regions use a nonlinear device kernel.

## Reports and compilation cache

```julia
details = report(fitted).details
```

Accelerator reports distinguish:

- requested and actual backend and device;
- accelerated, host, and host-fit nodes;
- last input/output residency, transfer locations, and estimated host byte counts;
- compilation and last-execution time;
- compilation count, cache hits, size, capacity, and evictions;
- portable-model and cache-wrapper host-memory estimates, with unavailable
  executable and peak device measurements explicitly marked `missing`;
- unsupported and fallback operations;
- whether the logistic objective was accelerated.

A `CompilationCache` belongs to `FitContext` and is shared by contexts derived
from it. Element types, parameter shapes, feature counts, and all input
dimensions (including batch size) are currently static compilation-signature
components. Different batch sizes therefore specialize separately. The cache
uses bounded least-recently-used eviction; its default capacity is 32 and can
be configured with `CompilationCache(max_entries=...)`.
Call `empty!(context.cache)` to release all cached executables explicitly;
subsequent operations compile the required entries again. Fitted parameters
remain in the portable model and are never retained by cache entries.

## Persistence behavior

Saving a fitted Reactant graph stores its authoritative CPU fitted graph. This
keeps model artifacts independent of a particular device runtime. Reloading
returns a portable `FittedGraph` with the same fitted parameters and predictions;
compiled executables, device arrays, locks, and accelerator cache state are not
serialized. Clearing or discarding the original cache cannot affect the loaded
model.

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
