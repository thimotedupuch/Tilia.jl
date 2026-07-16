# Reactant acceleration: limitations and implementation roadmap

## Status

Tilia's Reactant extension is currently a proof of concept, not a general
accelerated backend. It demonstrates that a fitted semantic pipeline can call a
compiled Reactant probability kernel, preserve CPU reference behavior, report
transfers and compilation, and fail or fall back explicitly.

The supported path is intentionally narrow:

```julia
Chain(Standardize(), LogisticRegression())
```

with a dense `Float32` or `Float64` matrix. Fitting still occurs on CPU. The
extension compiles inference and evaluates a compiled logistic objective, but
does not run the training solver on the device.

This document records the most important limitations visible in the current
implementation and proposes a coherent order of work. It should be updated as
each acceptance criterion becomes true.

## Most important current limitations

### 1. The compilation cache can reuse another fitted model's parameters

This is the most urgent correctness issue.

`_compile_for!` keys the shared `CompilationCache` using operation version,
input type and shape, feature count, and coefficient shape. The cached entry
contains both the compiled function and device copies of the first fitted
model's means, scales, coefficients, and intercept. A second model with the
same shapes and the same shared cache can therefore receive the first model's
bound parameters.

The cache must store compiled programs separately from model parameters. Every
execution must pass or bind the parameters belonging to the current fitted
object. Until that is fixed, sharing a `CompilationCache` between distinct
Reactant fits with compatible shapes is unsafe.

### 2. Backend support is a hard-coded whole-graph special case

`_supported` accepts exactly two semantic nodes: `Standardize` followed by
`LogisticRegression`. It does not use the existing numerical execution graph,
primitive lowering, device placement, or general node contracts to decide what
can run on Reactant.

Consequences include:

- no standalone logistic model;
- no additional compatible transforms before the classifier;
- no branches, selections, concatenation, or column mapping;
- no partial acceleration of a larger graph;
- no reusable per-operation lowering registry;
- duplicated knowledge between core graph contracts and the extension.

### 3. Training is not accelerated

The CPU graph is fitted before the Reactant wrapper is constructed. Means,
variances, logistic statistics, gradients, Hessians, and Newton updates remain
on CPU. `_compile_objective` evaluates the final objective once on device but
does not participate in optimization.

The report correctly records `host_fit_nodes`, but the current feature should
be described as compiled inference with an objective experiment—not accelerated
model fitting.

### 4. Host/device ownership is inefficient

Inference always:

1. materializes `Matrix(X)` during compilation preparation;
2. converts each new input with `Reactant.to_rarray`;
3. returns the result to the host with `Array(device_result)`.

`predict` calls `predict_proba` and therefore transfers the complete
probability matrix before choosing classes on CPU. There is no public path for
device-resident input, device-resident output, chained device operations, or
batched prediction without repeated transfers.

Model parameters are copied to the device for a compiled entry, but their
lifetime is coupled to the problematic shared cache rather than the fitted
model.

### 5. Every new observation shape compiles another program

The cache key contains `size(X)`, so different batch sizes compile separately.
The tests explicitly exercise a second row count, which currently implies a
second compilation. There is no dynamic-batch policy, bounded specialization
strategy, cache eviction, or clear distinction between shape-polymorphic and
shape-specialized operations.

### 6. Data and numerical coverage is very narrow

The extension accepts only dense `Float32` and `Float64` matrices. It does not
support:

- `ColumnTable` or general Tables.jl input at the accelerated boundary;
- sparse matrices;
- missing or categorical values;
- most transformations and model families;
- device-side class selection;
- incremental prediction buffers;
- a documented policy for mixed input and parameter floating types.

Observation weights reach the compiled objective experiment, but they do not
make fitting device-resident.

### 7. Device selection relies on mutable global state

When a non-`:auto` device is requested, fitting calls
`Reactant.set_default_backend`. Changing a process-wide default during a fit is
unsafe for concurrent fits, libraries sharing Reactant, and tests that assume
device isolation.

Device/backend selection should be carried by compiled objects or an explicit
Reactant context wherever the Reactant API allows it. If global mutation is
unavoidable, it needs synchronization, restoration, and documentation.

### 8. Diagnostics are useful but approximate

`transferred_bytes` uses `Base.summarysize` of host objects rather than measured
transfer payloads. Compilation and execution timers do not provide a complete
breakdown of host conversion, dispatch, synchronization, device execution, and
result materialization. `accelerator_nodes` describes inference eligibility,
while fitting of those same nodes occurred on the host; readers must combine
it with `host_fit_nodes` to understand the actual path.

Diagnostics need phase-specific node placement and clearly defined timing and
byte semantics before they can support performance decisions.

### 9. Test and benchmark coverage is insufficient for a backend

The isolated accelerator suite currently covers:

- one small binary `Float32` pipeline on Reactant CPU;
- probability and class agreement with CPU;
- cache hits for repeated calls;
- a second batch shape;
- explicit error and CPU fallback for one unsupported graph.

It does not cover Float64, multiclass inference, weights, GPU execution,
multiple fitted models sharing a cache, concurrency, parameter changes, empty
or large batches, feature mismatch beyond one path, compilation failure
injection, device selection isolation, or transfer accounting.

The benchmark is a Reactant-CPU snapshot for one model family. It does not
separate transfer-only cost, device kernel time, compilation amortization,
memory use, throughput across batch sizes, or GPU behavior.

## Implementation roadmap

## P0 — establish correctness and honest boundaries

These items should be completed before expanding model coverage.

### P0.1 Separate executable caching from fitted parameters

Refactor cache entries so they contain only reusable compiled code and
shape/type/backend metadata. Store device parameters on the fitted Reactant
object or pass current parameters on every call.

Acceptance criteria:

- two differently fitted models with identical shapes share a compilation
  cache and still match their respective CPU predictions;
- changing coefficients, intercept, means, or scales cannot reuse stale values;
- cache keys include operation/lowering version, backend/device, relevant
  numerical policy, element types, and required static shapes;
- cache behavior has direct regression tests.

### P0.2 Define phase-specific backend capability queries

Replace the two-node `_supported` predicate with capability checks for
individual lowered operations and phases (`:fit`, `:predict`, and
`:predict_proba`). The existing `NumericalExecutionGraph` should be the common
input to backend support decisions.

Acceptance criteria:

- support is reported per numerical node or primitive;
- unsupported reasons identify the exact operation and phase;
- semantic graph contracts and Reactant lowering cannot disagree silently;
- standalone supported nodes no longer require a synthetic two-node pattern.

### P0.3 Make reports unambiguous

Report fit and inference placement separately. Define whether each timer
includes conversion and synchronization, and distinguish estimated host bytes
from actual device transfer bytes.

Acceptance criteria:

- reports answer which nodes fitted on CPU, which inference nodes ran on the
  device, and where transfers occurred;
- compilation, host conversion, device execution, synchronization, and result
  materialization have documented fields or documented aggregation;
- CPU fallback reports preserve the requested backend and exact reason;
- report tests validate field semantics, not only nonzero values.

### P0.4 Expand correctness tests before performance work

Add Float32/Float64, binary/multiclass, weighted objective, distinct-model cache,
multiple batch shapes, empty-batch policy, large batch, concurrency, explicit
device, compilation failure, and fallback tests. GPU tests may be conditional
on CI hardware, but the test contract should exist.

## P1 — build a reusable inference backend

Once P0 is complete, broaden inference through small composable operations
rather than adding another hard-coded model pattern.

### P1.1 Implement a lowering registry for common primitives

Prioritize operations already represented in the numerical graph:

1. affine transforms: standardize, min--max scaling, imputation of numeric
   constants, linear projection;
2. dense primitives: matrix multiplication, bias addition, reductions,
   sigmoid, softmax/log-softmax, argmax;
3. composition: select/gather, concatenate, and compatible sequential regions;
4. prediction heads: linear/ridge regression and binary/multiclass logistic
   regression;
5. decomposition inference: PCA and truncated-SVD transforms.

Each lowering should have CPU-reference tests for types, shapes, edge values,
and numerical tolerance.

### P1.2 Partition mixed graphs

Use device placement to form maximal supported Reactant regions separated by
explicit transfers. `fallback=:cpu` should permit unsupported regions to run on
CPU without abandoning acceleration for all supported regions.

Acceptance criteria:

- a graph with supported and unsupported nodes executes both regions correctly;
- transfer nodes appear in the numerical execution graph and report;
- adjacent supported nodes stay device-resident;
- fallback does not rebuild or refit the pipeline;
- results match authoritative CPU execution.

### P1.3 Introduce device-resident prediction paths

Keep fitted parameters with the fitted model, support device arrays as input
where practical, and allow a caller to request host or device output. Implement
class selection on device so `predict` need not transfer a full probability
matrix.

The public API should remain backend-neutral; device-specific controls belong
in `FitContext`, an execution option, or an explicitly documented optional API.

### P1.4 Define a shape specialization policy

Investigate Reactant support for dynamic leading batch dimensions. If dynamic
batches are not viable, define bounded specialization and cache eviction.

Acceptance criteria:

- batch-size changes do not cause unbounded cache growth;
- compile count and cache size are reported;
- documentation states which dimensions are static;
- steady-state benchmarks use already compiled signatures.

## P2 — accelerate fitting coherently

Fitting should be added only after inference uses the general lowering and
placement infrastructure.

### P2.1 Device-resident sufficient statistics

Start with standardization reductions and linear/logistic sufficient
statistics. Preserve Tilia's accumulation type, deterministic-reduction, and
weight semantics. Compare results against CPU for ill-scaled and weighted data,
not only small well-conditioned matrices.

### P2.2 Accelerated logistic optimization

Move objective, gradient, Hessian or Hessian-vector operations, and solver
iterations into a coherent device loop. Avoid a design that transfers a
gradient or Hessian on every iteration unless benchmarks justify it.

Acceptance criteria:

- fitting produces the documented objective and stopping behavior;
- convergence and iteration reports agree in meaning with CPU reports;
- weighted binary and multiclass one-vs-rest paths are tested;
- numerical failures surface as typed Tilia errors;
- device fitting is measurably beneficial beyond compilation cost for a
  documented workload range.

### P2.3 Add model families by shared primitives

After logistic training, prioritize models that reuse established operations:

1. linear and ridge regression;
2. PCA/truncated-SVD transform and, if supported well by Reactant, fitting;
3. SGD linear models with device-resident batches;
4. shallow MLP inference and later fitting.

Tree algorithms, sparse coordinate descent, DBSCAN, nearest-neighbor search,
and complex categorical preprocessing should not be early targets unless a
clear Reactant lowering and performance case exists.

## P3 — production readiness

### Device and concurrency safety

- remove unsynchronized global backend mutation;
- support concurrent fitted objects and predictions;
- document thread safety of caches and device buffers;
- test CPU and available GPU backends in isolated processes.

### Memory and lifecycle management

- account for device parameter, executable, input, and output memory;
- provide bounded cache eviction and cleanup;
- avoid retaining model parameters only because a compilation cache survives;
- benchmark peak host and device memory.

### Benchmark matrix

Measure separately:

- first compilation;
- first execution after compilation;
- steady-state kernel execution;
- host-to-device and device-to-host transfer;
- end-to-end latency and throughput;
- cache reuse across batch sizes and fitted models;
- CPU and GPU where available;
- Float32 and Float64;
- small, medium, and large observation/feature regimes.

Every published comparison should include hardware, Julia, Reactant, Tilia,
threading, batch shape, warm-up, synchronization, and whether transfer time is
included.

### Persistence and deployment

Continue saving the authoritative CPU graph by default. If compiled artifact
persistence is later considered, it must be explicitly versioned by platform,
Reactant/XLA version, device target, shapes, and lowering version, with a safe
fallback to recompilation. Portable fitted parameters and non-portable compiled
executables should remain separate concerns.

## Recommended next three pull requests

1. **Cache correctness:** separate compiled executable reuse from fitted device
   parameters and add cross-model shared-cache regression tests.
2. **General inference lowering:** route Standardize and logistic inference
   through `NumericalExecutionGraph` capability/lowering records instead of the
   exact two-node `_supported` predicate.
3. **Mixed-region execution:** implement explicit CPU/Reactant partitioning and
   phase-specific reporting for a graph containing at least one supported and
   one unsupported region.

These three changes establish a trustworthy foundation. Expanding the model
catalog before them would multiply special cases while leaving correctness,
placement, and reporting unresolved.

## Relevant implementation locations

- `ext/TiliaReactantExt/lowering.jl`: current whole-graph support predicate;
- `ext/TiliaReactantExt/compile.jl`: compiled probability and objective kernels;
- `ext/TiliaReactantExt/cache.jl`: current executable/parameter cache coupling;
- `ext/TiliaReactantExt/backend.jl`: fitted wrapper and host/device execution;
- `ext/TiliaReactantExt/diagnostics.jl`: report and fallback semantics;
- `src/graph/execution_plan.jl`: numerical lowering and buffer planning;
- `src/graph/passes/device_placement.jl`: backend placement and transfers;
- `test/accelerator/runtests.jl`: isolated accelerator correctness tests;
- `benchmark/accelerator/`: current Reactant CPU benchmark and snapshot report.
