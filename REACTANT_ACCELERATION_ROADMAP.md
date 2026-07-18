# Reactant acceleration: limitations and implementation roadmap

## Status

Tilia's Reactant extension now provides numerical capability checks,
mixed-device inference regions, device-resident prediction, bounded
specialization caching, and initial accelerated fitting for dense `Float32`
and `Float64` pipelines. CPU implementations remain the reference,
unsupported work is explicit, and persistence stores the portable CPU graph.
Logistic Newton and regularized ridge Cholesky fitting are device-resident for
their documented supported graph shapes.

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

### 3. Training acceleration status

Supported standardization, weighted ridge sufficient statistics, and logistic
Newton optimization now execute through Reactant. Unsupported fit primitives
remain explicitly reported in `host_fit_nodes`; supported linear logistic
and ridge graphs construct portable placeholders and no longer perform duplicate
CPU transform or solver fits before device fitting.

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

Observation weights participate in device-resident logistic and ridge fitting
for the documented supported graphs.

### 7. Device selection uses synchronized mutable global state

When a non-`:auto` device is requested, fitting calls
`Reactant.set_default_backend`. Tilia serializes every Reactant fit and
prediction through a process-local reentrant lock and restores the previous
backend in `finally`. Concurrent fitted objects, shared-cache fits, parameter
isolation, and exact backend restoration are tested.

An explicit per-compiled-object Reactant context remains preferable if the API
supports one in the future. Tilia's lock cannot coordinate unrelated external
code that mutates Reactant's global backend directly.

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

**Completed.** The shared cache stores only compiled executables. Current
fitted parameters are supplied on every execution, and the cache signature
includes lowering version, requested and resolved backend, numerical policy,
element types, and static shapes. Regression tests cover distinct fitted
models sharing a cache and parameter changes after compilation.

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

**Completed.** Reactant capability decisions consume the backend-neutral
`NumericalExecutionGraph` and record support per lowered primitive and phase.
Unsupported errors identify the primitive and phase, fit and inference
capabilities are retained in the report, and a standalone logistic numerical
graph is supported without a preceding synthetic transform.

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

**Completed.** Reports contain explicit fit and inference placement, separate
compilation, host conversion, device execution, and synchronization/result
materialization timing buckets, and transfer accounting labeled as a
`summarysize`-based host estimate with actual device bytes reported as
unavailable. CPU fallback retains the requested Reactant backend and reason.

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

**Completed.** Coverage includes Float32 and Float64, binary and
multiclass inference, weighted objective agreement, distinct-model shared
caches, multiple and large batch shapes, an explicit empty-batch error policy,
feature mismatch, concurrent predictions, explicit CPU device selection, and
fallback behavior. A hardware-conditional contract validates GPU probability
agreement when available and compilation-failure fallback otherwise. Explicit
backend selection is synchronized and restores the prior Reactant backend.

Add Float32/Float64, binary/multiclass, weighted objective, distinct-model cache,
multiple batch shapes, empty-batch policy, large batch, concurrency, explicit
device, compilation failure, and fallback tests. GPU tests may be conditional
on CI hardware, but the test contract should exist.

## P1 — build a reusable inference backend

Once P0 is complete, broaden inference through small composable operations
rather than adding another hard-coded model pattern.

### P1.1 Implement a lowering registry for common primitives

**Completed.** A model/primitive lowering registry composes linear
regions from identity, standardization, unclipped min--max scaling, PCA
(including whitening), and truncated SVD with binary/multiclass logistic and
linear/ridge prediction heads. Transform parameters are folded into generic
effective coefficient and intercept arrays rather than retained in a
model-pattern-specific executable. CPU-reference tests cover standalone and
composed Float32 regression/classification paths, decomposition projections,
clipped min--max values outside the fitted range, and binary/multiclass
device-side class selection with first-index tie behavior. Numeric imputation
is lowered as a device `select_fill` using boundary-encoded values and a
missing mask. DAG-aware lowering folds select/gather and concatenate branches
into effective prediction maps, with CPU-reference coverage for parallel
standardization/PCA regions.

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

**Completed.** With `fallback=:cpu`, graphs are partitioned into
maximal supported Reactant regions separated by CPU transforms. The
authoritative CPU graph is fitted once; transform-only Reactant regions,
device-to-host transitions, CPU nodes, host-to-device transitions, and the
final prediction region execute in order. Placement and both transfer
directions appear on numerical primitives and in phase-specific reports.
Tests cover both a CPU prefix followed by an adjacent Reactant suffix and two
Reactant regions separated by RobustScale, with CPU agreement and fitted-node
identity across predictions. Branched DAG coverage accelerates one branch and
the final head around a CPU branch/concatenation, deriving transfers from real
graph edges. Clipped min--max transform regions cover nonlinear device regions
separated from other supported regions by a CPU transform.

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

**Completed for supported whole-graph inference.** Classification `predict` compiles binary selection or
multiclass first-index argmax into the Reactant program and transfers only the
resulting index vector before mapping indices to arbitrary host class labels.
Concrete Reactant matrices are accepted without re-uploading the input, and
`predict_proba` plus regression `predict` accept `output=:host` or `:device`.
Classification labels intentionally remain host-side because their Julia types
may not be device-representable. Reports expose last input/output residency,
materialization timing, and estimated transfers. Mixed graphs still materialize
at required CPU boundaries.

Keep fitted parameters with the fitted model, support device arrays as input
where practical, and allow a caller to request host or device output. Implement
class selection on device so `predict` need not transfer a full probability
matrix.

The public API should remain backend-neutral; device-specific controls belong
in `FitContext`, an execution option, or an explicitly documented optional API.

### P1.4 Define a shape specialization policy

**Completed.** All dimensions, including batch size, are documented static
signature components. `CompilationCache` has configurable bounded LRU-style
eviction (default capacity 32), and reports compilation count, cache hits,
current size, capacity, and evictions. Regression tests specialize across
eight batch sizes and prove bounded growth. Accelerator benchmarks warm the
exact signature before steady-state samples and print the specialization and
cache policy.

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

**Completed for the prioritized statistics.** Eligible linear graphs beginning with `Standardize` compile
population mean and second-moment reductions on Reactant using the configured
accumulation type. That fitted transform is authoritative, and downstream
nodes are refit once against its output rather than retaining coefficients
from shadow CPU statistics. Fit placement, timings, transfer estimates,
accumulation type, and deterministic-reduction policy are reported. Tests cover
ill-scaled `Float32` input with `Float64` accumulation and a weighted downstream
ridge fit. Weighted ridge now computes centering, Gram and cross-product
statistics, the regularized Cholesky solve, intercept recovery, and residual
norm in one Reactant program; centered/intercepted and raw/no-intercept paths
are tested. Logistic gradient and Hessian statistics now remain device-side
inside the coherent Newton optimizer. QR/SVD linear regression deliberately remains on CPU because a
normal-equation substitution would change solver semantics. Logistic
sufficient-statistic transfers are therefore avoided.

Start with standardization reductions and linear/logistic sufficient
statistics. Preserve Tilia's accumulation type, deterministic-reduction, and
weight semantics. Compare results against CPU for ill-scaled and weighted data,
not only small well-conditioned matrices.

### P2.2 Accelerated logistic optimization

**Completed.** Binary and multiclass one-vs-rest logistic fits run a coherent
Reactant Newton program: stable sigmoid/objective evaluation, weighted
gradient, Hessian construction, Cholesky solve, convergence masking, and all
21 Armijo candidates execute without per-iteration host transfers. The Armijo
candidates are evaluated as one batched tensor and the largest acceptable
scale reproduces the CPU halving order. Final parameters, convergence,
iteration count, gradient norm, and the complete bounded objective history
materialize once per class. Weighted binary and three-class CPU-reference tests
pass. Workload-range performance evidence remains part of the broader
acceleration benchmark gate.

Current Reactant-CPU evidence (Julia 1.12.4, Reactant 0.2.273, 16 features,
one warm-fit sample) is intentionally negative: at 100 observations the warm
fit speedup is `0.00088×`, and at 1,000 observations it is `0.069×`. Steady-state
prediction takes about 119 μs and 161 μs respectively, versus CPU's 4.9 μs and
34.7 μs. These figures document that Reactant CPU is a correctness/development
target for these shapes, not a performance claim. The acceptance gate remains
open pending measurements on a supported accelerator and a beneficial workload
range.

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

**In progress.** Ridge regression with positive regularization and the
documented Cholesky solver now has device-resident weighted fitting for both
intercepted and raw paths. It reuses reduction and Cholesky primitives already
established by P2.1 and P2.2, avoids a shadow CPU ridge solve, and reports
`solver_backend=:reactant`. General
linear regression, decomposition fitting, SGD models, and MLP fitting remain.

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

**Completed within the current Reactant backend-selection API.** Global backend
selection is synchronized and restored, fitted-object execution is serialized,
compilation caches are locked and bounded, and concurrent fits/predictions are
covered. CPU is tested in-process; unavailable GPU targets retain explicit
fallback/error coverage. Truly isolated CPU/GPU CI remains deployment work.

- retain synchronization until Reactant exposes per-object backend contexts;
- test available GPU backends in isolated CI processes.

### Memory and lifecycle management

**Partially completed.** Compilation caches are bounded LRU stores containing
executables only, expose thread-safe `empty!` cleanup, and report an atomic
host-retained size estimate. Reports separate the portable model, compilation
cache wrappers, retained device parameters (zero), and unavailable executable
or peak device measurements instead of presenting transfer estimates as memory
usage. Measured peak host/device memory remains pending.

- account for measured device executable, input, and output memory;
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

**Completed for portable model persistence.** Saving a Reactant-fitted graph
serializes only its authoritative `FittedGraph`; compiled executables, device
arrays, locks, and cache state are excluded. Accelerator tests load the artifact
as a CPU-portable graph, verify exact parameter/prediction round trips, and show
that clearing the original cache cannot affect the loaded model.

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
