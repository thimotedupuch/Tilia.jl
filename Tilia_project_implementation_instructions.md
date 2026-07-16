# Tilia.jl — Implementation Instructions

## 1. Project objective

Build a unified, Julia-native classical machine-learning stack with:

- one repository;
- one primary package;
- one public API;
- one release cycle;
- minimal hard dependencies;
- native implementations of algorithms;
- an internal graph execution model;
- optional accelerator and differentiation integrations.

The project is not intended to be an adapter layer over the existing Julia machine-learning ecosystem. Existing packages may be used as references for algorithms, validation, and numerical behavior, but the production implementation should be owned by this project.

The initial goal is not to match the entire breadth of scikit-learn. The first goal is to establish the correct architecture, semantics, and testing discipline so that models can be added incrementally without fragmenting the system.

---

## 2. Non-goals

Do not begin by attempting to:

- implement every algorithm;
- wrap existing packages behind a common interface;
- create a plugin ecosystem;
- make all algorithms accelerator-compatible;
- make automatic differentiation mandatory;
- reproduce scikit-learn's API exactly;
- build a general-purpose compiler;
- support arbitrary Julia objects as pipeline nodes;
- optimize every operation before the semantics are stable.

The project should prefer a smaller coherent system over broad but inconsistent feature coverage.

---

## 3. Core design principles

### 3.1 One coherent implementation

Models, preprocessing, metrics, model selection, persistence, data handling, and graph execution should be implemented in the repository.

Avoid delegating core behavior to many independently versioned packages.

### 3.2 Minimal dependencies

A dependency is acceptable only when it provides infrastructure that is:

- difficult or risky to reimplement;
- broadly useful across the codebase;
- stable and narrowly scoped;
- not itself a machine-learning framework.

### 3.3 Explicit semantics

The project must standardize:

- observation orientation;
- class ordering;
- prediction shapes;
- missing-value behavior;
- categorical behavior;
- weighting semantics;
- random-number generation;
- numerical precision;
- convergence reporting;
- serialization;
- device fallback behavior.

Do not leave these behaviors implicit or model-specific.

### 3.4 Separation of specification and fitted state

Unfitted model definitions are immutable hyperparameter specifications.

Fitted objects are separate values containing learned state and reports.

### 3.5 Graph-based execution

Pipelines should be represented as semantic graphs, then lowered into numerical execution graphs.

The graph should support:

- validation;
- schema propagation;
- leakage prevention;
- device placement;
- graph optimization;
- reproducibility;
- introspection;
- accelerator compilation.

### 3.6 Progressive acceleration

CPU support is the baseline.

GPU, TPU, and other accelerator support should be added where it provides real benefits. Unsupported or partially supported execution must be explicit.

### 3.7 Testing before breadth

Every algorithm must pass shared conformance tests and mathematical invariant tests before being considered complete.

---

## 4. Dependency policy

## 4.1 Julia standard libraries

Use Julia standard libraries wherever possible.

Expected standard-library dependencies:

```toml
[deps]
LinearAlgebra = "..."
SparseArrays = "..."
Statistics = "..."
Random = "..."
Logging = "..."
Printf = "..."
Dates = "..."
TOML = "..."
SHA = "..."
UUIDs = "..."
Mmap = "..."
Serialization = "..."
Distributed = "..."
```

Primary roles:

- `LinearAlgebra`: BLAS, LAPACK, decompositions, factorizations, eigensystems.
- `SparseArrays`: sparse matrices and SuiteSparse-backed operations.
- `Statistics`: common descriptive statistics.
- `Random`: deterministic random streams and sampling.
- `Logging`: structured diagnostics.
- `TOML`: metadata, configuration, and persistent manifests.
- `SHA`: cache keys and integrity checks.
- `UUIDs`: persistent graph and artifact identities when required.
- `Mmap`: large model arrays and datasets.
- `Serialization`: temporary caches only.
- `Distributed`: process-level parallel execution.
- `Base.Threads`: default CPU parallelism.

## 4.2 Hard external dependencies

Initially allow only:

```toml
[deps]
Tables = "..."
SpecialFunctions = "..."
```

### Tables.jl

Use `Tables.jl` as the tabular interoperability boundary.

Do not depend directly on:

- DataFrames;
- CSV;
- Arrow;
- TypedTables;
- CategoricalArrays.

Convert any compatible table into the project's own internal representation.

### SpecialFunctions.jl

Use it for numerically reliable special functions such as:

- `loggamma`;
- `digamma`;
- error functions;
- incomplete gamma functions;
- functions needed by statistical likelihoods.

Do not reimplement special functions.

## 4.3 Weak dependencies

Use Julia package extensions for:

```toml
[weakdeps]
Reactant = "..."
DifferentiationInterface = "..."

[extensions]
TiliaReactantExt = "Reactant"
TiliaDifferentiationExt = "DifferentiationInterface"
```

These integrations remain in the same repository.

Makie recipes live in the separate `TiliaMakieRecipes` package and must not be
a Tilia weak dependency or extension.

## 4.4 Dependencies to avoid

Do not make the main package depend on:

- MLJ;
- MLJModelInterface;
- StatsBase;
- StatsAPI;
- Optim;
- Distributions;
- GLM;
- MultivariateStats;
- Clustering;
- NearestNeighbors;
- DecisionTree;
- Flux;
- NNlib;
- MLUtils;
- DataFrames;
- CategoricalArrays;
- CUDA;
- AMDGPU;
- plotting frameworks;
- experiment-tracking frameworks;
- model-serialization frameworks.

Developer-only dependencies are acceptable in isolated environments.

---

## 5. Repository structure

Use one monorepo and one package.

```text
Tilia.jl/
├── Project.toml
├── Manifest-v1.x.toml
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── src/
│   ├── Tilia.jl
│   │
│   ├── core/
│   │   ├── api.jl
│   │   ├── estimator.jl
│   │   ├── fitted.jl
│   │   ├── traits.jl
│   │   ├── context.jl
│   │   ├── reports.jl
│   │   ├── errors.jl
│   │   └── numerics_policy.jl
│   │
│   ├── data/
│   │   ├── schema.jl
│   │   ├── dataset.jl
│   │   ├── table_adapter.jl
│   │   ├── column_table.jl
│   │   ├── categorical.jl
│   │   ├── missing.jl
│   │   ├── batches.jl
│   │   ├── views.jl
│   │   └── device_transfer.jl
│   │
│   ├── graph/
│   │   ├── graph.jl
│   │   ├── node.jl
│   │   ├── ports.jl
│   │   ├── state.jl
│   │   ├── builder.jl
│   │   ├── validation.jl
│   │   ├── schema_inference.jl
│   │   ├── shape_inference.jl
│   │   ├── lowering.jl
│   │   ├── execution_plan.jl
│   │   ├── interpreter.jl
│   │   ├── cache.jl
│   │   └── passes/
│   │       ├── constant_folding.jl
│   │       ├── dead_node_elimination.jl
│   │       ├── transform_fusion.jl
│   │       ├── device_placement.jl
│   │       ├── buffer_planning.jl
│   │       └── leakage_validation.jl
│   │
│   ├── kernels/
│   │   ├── reductions.jl
│   │   ├── normalization.jl
│   │   ├── logexp.jl
│   │   ├── distances.jl
│   │   ├── pairwise.jl
│   │   ├── covariance.jl
│   │   ├── ranking.jl
│   │   ├── selection.jl
│   │   ├── sampling.jl
│   │   ├── sparse.jl
│   │   ├── categorical.jl
│   │   ├── losses.jl
│   │   └── activations.jl
│   │
│   ├── solvers/
│   │   ├── solver.jl
│   │   ├── convergence.jl
│   │   ├── line_search.jl
│   │   ├── least_squares.jl
│   │   ├── conjugate_gradient.jl
│   │   ├── newton.jl
│   │   ├── newton_cg.jl
│   │   ├── lbfgs.jl
│   │   ├── coordinate_descent.jl
│   │   ├── proximal_gradient.jl
│   │   ├── fista.jl
│   │   ├── stochastic.jl
│   │   └── em.jl
│   │
│   ├── preprocessing/
│   │   ├── standardize.jl
│   │   ├── normalize.jl
│   │   ├── impute.jl
│   │   ├── encode.jl
│   │   ├── discretize.jl
│   │   ├── polynomial.jl
│   │   ├── feature_hashing.jl
│   │   ├── column_selection.jl
│   │   └── composition.jl
│   │
│   ├── models/
│   │   ├── linear/
│   │   ├── decomposition/
│   │   ├── clustering/
│   │   ├── neighbors/
│   │   ├── naive_bayes/
│   │   ├── discriminant/
│   │   ├── mixture/
│   │   ├── trees/
│   │   ├── ensembles/
│   │   ├── kernels/
│   │   ├── svm/
│   │   ├── manifold/
│   │   ├── outliers/
│   │   ├── neural/
│   │   └── semi_supervised/
│   │
│   ├── metrics/
│   │   ├── classification.jl
│   │   ├── regression.jl
│   │   ├── ranking.jl
│   │   ├── clustering.jl
│   │   ├── pairwise.jl
│   │   └── curves.jl
│   │
│   ├── model_selection/
│   │   ├── split.jl
│   │   ├── resampling.jl
│   │   ├── cross_validation.jl
│   │   ├── evaluation.jl
│   │   ├── search_space.jl
│   │   ├── grid_search.jl
│   │   ├── random_search.jl
│   │   ├── successive_halving.jl
│   │   └── result.jl
│   │
│   ├── inspection/
│   │   ├── confusion_matrix.jl
│   │   ├── calibration.jl
│   │   ├── learning_curve.jl
│   │   ├── partial_dependence.jl
│   │   ├── permutation_importance.jl
│   │   └── diagnostics.jl
│   │
│   ├── persistence/
│   │   ├── format.jl
│   │   ├── manifest.jl
│   │   ├── array_storage.jl
│   │   ├── save.jl
│   │   ├── load.jl
│   │   └── migrations.jl
│   │
│   └── registry/
│       ├── registry.jl
│       ├── capabilities.jl
│       └── discovery.jl
│
├── ext/
│   ├── TiliaReactantExt/
│   └── TiliaDifferentiationExt/
│
├── test/
│   ├── runtests.jl
│   ├── core/
│   ├── data/
│   ├── graph/
│   ├── kernels/
│   ├── solvers/
│   ├── models/
│   ├── metrics/
│   ├── model_selection/
│   ├── persistence/
│   ├── conformance/
│   ├── numerical/
│   ├── regression/
│   ├── allocation/
│   └── accelerator/
│
├── benchmark/
│   ├── kernels/
│   ├── training/
│   ├── inference/
│   ├── pipelines/
│   └── compilation/
│
├── docs/
│   ├── make.jl
│   └── src/
│
└── scripts/
    ├── generate_reference_data.jl
    ├── update_model_manifest.jl
    └── check_public_api.jl
```

Do not split these directories into separate packages until there is a demonstrated technical need.

---

## 6. Public API

The public API should remain small and stable.

Initial exported functions:

```julia
fit
predict
predict_proba
transform
inverse_transform
partial_fit
evaluate
tune
report
save_model
load_model
```

Initial exported composition types:

```julia
Chain
Parallel
ColumnMap
Select
Concatenate
```

Initial execution types:

```julia
CPUBackend
ReactantBackend
FitContext
```

Initial result types:

```julia
ConfusionMatrix
ROCResult
CrossValidationResult
OptimizationTrace
```

Internal solver, graph, compiler, and storage types should not be exported by default.

Advanced namespaces may be exposed as:

```julia
Tilia.Graph
Tilia.Kernels
Tilia.Solvers
Tilia.Experimental
```

---

## 7. Core type contracts

## 7.1 Model specifications

Models are immutable hyperparameter descriptions.

Example:

```julia
struct LogisticRegression{T,S}
    penalty::Symbol
    λ::T
    solver::S
    fit_intercept::Bool
end
```

Constructors must validate hyperparameters.

Do not store training data or learned parameters in model specifications.

## 7.2 Fitted objects

Training returns a separate fitted type.

Example:

```julia
struct FittedLogisticRegression{M,C,I,R}
    model::M
    coefficients::C
    intercept::I
    report::R
end
```

The fitted object must contain:

- the original model specification;
- learned parameters;
- data-dependent metadata;
- schema information;
- a structured report.

## 7.3 Training context

Execution configuration belongs in a context.

Example:

```julia
struct FitContext{B,R,N,C}
    backend::B
    rng::R
    numerics::N
    cache::C
end
```

The context may include:

- backend;
- random stream;
- numeric precision policy;
- deterministic execution flag;
- logging policy;
- memory budget;
- compilation cache;
- threading policy.

Models should not contain device configuration unless it is a statistical hyperparameter.

## 7.4 Traits and capabilities

Define explicit capabilities.

Examples:

```julia
task_kind(::Type{<:LogisticRegression}) = ClassificationTask()
supports_sparse(::Type{<:LogisticRegression}) = true
supports_missing(::Type{<:LogisticRegression}) = false
supports_weights(::Type{<:LogisticRegression}) = true
supports_partial_fit(::Type{<:LogisticRegression}) = false
is_probabilistic(::Type{<:LogisticRegression}) = true
```

Capabilities must be used for validation, reporting, and model discovery.

Do not rely on runtime method errors for expected incompatibilities.

---

## 8. Data system

## 8.1 Accepted inputs

The public API should accept:

- `AbstractMatrix`;
- `Tables.jl`-compatible tables;
- native `Dataset` objects.

Convert external data into native structures before fitting.

## 8.2 Observation orientation

Standardize on rows as observations and columns as features.

This convention must apply to:

- matrices;
- tables;
- predictions;
- metrics;
- model reports;
- graph schemas.

Any algorithm that is more naturally column-oriented must adapt internally.

## 8.3 Native dataset

Define a native container such as:

```julia
struct Dataset{X,Y,W,S}
    features::X
    target::Y
    weights::W
    schema::S
end
```

Support:

- no target;
- no weights;
- dense and sparse features;
- tabular and matrix features;
- training and inference schemas.

## 8.4 Native table representation

Use a column-oriented internal representation.

Example:

```julia
struct ColumnTable{N,C,S}
    names::N
    columns::C
    schema::S
end
```

Do not use DataFrames internally.

## 8.5 Categorical data

Implement an internal categorical representation.

Example:

```julia
struct CategoricalColumn{T,C,P}
    codes::C
    pool::P
end
```

The schema must record:

- ordered or unordered;
- known levels;
- unknown-level policy;
- missing-value policy;
- code element type.

## 8.6 Schema

A schema should track:

- column names;
- logical types;
- physical types;
- feature roles;
- missingness;
- category levels;
- feature provenance;
- generated feature names;
- target metadata;
- class ordering.

Schema propagation is a required part of graph construction.

---

## 9. Graph architecture

Use two graph levels.

## 9.1 Semantic model graph

Represents user-meaningful operations.

Examples:

- `SelectColumns`;
- `Impute`;
- `Standardize`;
- `OneHotEncode`;
- `Concatenate`;
- `PCA`;
- `LogisticRegression`.

Each node must declare:

- input contract;
- output schema rule;
- whether it learns state;
- whether it consumes the target;
- whether it changes row count;
- whether it changes feature count;
- whether it is valid at inference;
- sparse compatibility;
- missing-value compatibility;
- backend compatibility.

## 9.2 Numerical execution graph

Represents lower-level operations.

Examples:

- gather;
- scatter;
- reduction;
- matrix multiplication;
- factorization;
- normalization;
- indexing;
- device transfer;
- solver loop.

The graph must carry:

- shapes;
- element types;
- sparse or dense representation;
- device placement;
- buffer lifetime;
- aliasing information;
- mutability constraints.

## 9.3 Graph lifecycle

The expected lifecycle is:

```text
User specification
    ↓
Semantic graph construction
    ↓
Input validation
    ↓
Schema propagation
    ↓
Leakage validation
    ↓
Fit graph construction
    ↓
Lowering to numerical operations
    ↓
Optimization passes
    ↓
Backend selection
    ↓
CPU interpretation or accelerator compilation
    ↓
Fitted graph and structured report
```

Use separate fit and inference graphs.

## 9.4 Graph optimization passes

Initial passes:

- constant folding;
- dead-node elimination;
- transform fusion;
- redundant conversion elimination;
- device placement;
- transfer coalescing;
- buffer planning;
- leakage validation.

Optimization passes must preserve graph semantics.

Each pass should have unit tests based on before-and-after graph equivalence.

## 9.5 Leakage prevention

The graph must distinguish:

- fit-time computations;
- inference-time computations;
- target-dependent computations;
- train-only transformations.

Cross-validation must fit the entire pipeline independently within every fold.

Prevent fitting preprocessing on full data before splitting.

---

## 10. CPU execution model

Implement a CPU graph interpreter before compiler integration.

The interpreter should:

- execute nodes in topological order;
- allocate or reuse buffers;
- support dense and sparse arrays;
- support multithreaded kernels;
- emit structured timing data;
- allow debugging and graph tracing.

Graph operations should be semantically functional, even when the CPU planner lowers them to mutation.

Example semantic operation:

```julia
C = matmul(A, B)
```

Possible CPU lowering:

```julia
mul!(C_buffer, A, B)
```

Mutation must remain an execution optimization, not part of user-visible graph semantics.

---

## 11. Kernel layer

Centralize common numerical operations.

Initial kernels should include:

### Reductions

- sum;
- weighted sum;
- mean;
- weighted mean;
- variance;
- weighted variance;
- stable norm;
- min/max;
- argmin/argmax.

### Stable transforms

- log-sum-exp;
- softmax;
- log-softmax;
- sigmoid;
- stable binary cross-entropy;
- clipping.

### Distances

- Euclidean;
- squared Euclidean;
- Manhattan;
- cosine;
- Mahalanobis where applicable;
- pairwise distance blocks.

### Statistical kernels

- covariance;
- weighted covariance;
- contingency accumulation;
- class counting;
- histogram building.

### Sparse kernels

- sparse reductions;
- sparse scaling;
- sparse centering policies;
- sparse dot products;
- sparse matrix-vector operations.

### Selection and ranking

- top-k;
- partial sorting;
- quantile helpers;
- rank handling;
- tie handling.

Every kernel should have:

- a generic implementation;
- tests for correctness;
- tests for edge cases;
- allocation tests;
- a path for future Reactant lowering.

Do not duplicate kernel logic inside individual models.

---

## 12. Solver architecture

Separate:

- statistical model;
- objective function;
- regularizer;
- constraints;
- initialization;
- solver;
- stopping condition;
- execution backend.

Conceptual interface:

```julia
objective(model, parameters, batch)
initial_parameters(model, data, context)
solve(solver, objective, initial_parameters, context)
```

Initial solvers:

- QR least squares;
- SVD least squares;
- Cholesky least squares;
- conjugate gradient;
- LSQR or LSMR;
- Newton;
- Newton-CG;
- IRLS;
- L-BFGS;
- coordinate descent;
- proximal gradient;
- FISTA;
- stochastic gradient methods;
- expectation-maximization.

A model may bypass the generic solver layer when its algorithm is inherently specialized.

Examples:

- decision trees;
- exact nearest-neighbor index construction;
- histogram boosting;
- eigendecomposition-based PCA.

---

## 13. Reactant integration

Reactant should be an optional backend, not a public abstraction leaked through the codebase.

The extension should contain:

```text
ext/TiliaReactantExt/
├── TiliaReactantExt.jl
├── backend.jl
├── arrays.jl
├── lowering.jl
├── compile.jl
├── cache.jl
├── diagnostics.jl
└── ops/
    ├── reductions.jl
    ├── linear_algebra.jl
    ├── indexing.jl
    ├── sparse.jl
    └── control_flow.jl
```

Do not scatter checks such as:

```julia
if backend isa ReactantBackend
```

through model implementations.

Instead, lower numerical graph operations through backend dispatch.

The execution report must record:

- nodes placed on the accelerator;
- nodes remaining on the host;
- transfer locations;
- compilation time;
- execution time;
- transferred bytes;
- unsupported operations;
- fallbacks.

Fallback must never be silent.

---

## 14. Differentiation integration

Automatic differentiation is optional.

Core models should prefer:

1. closed-form solutions;
2. analytic gradients;
3. analytic Hessians or Hessian-vector products;
4. specialized algorithms;
5. automatic differentiation only when appropriate.

Define a backend-independent derivative protocol:

```julia
value(objective, parameters, data)
gradient!(destination, objective, parameters, data)
value_and_gradient!(destination, objective, parameters, data)
jvp(...)
vjp(...)
```

The DifferentiationInterface extension may provide derivatives for custom objectives or operations without analytic rules.

Do not attempt to differentiate arbitrary model-fitting control flow.

Differentiate numerical objectives and graph operations.

---

## 15. Makie recipes

The core should produce semantic result types.

Examples:

```julia
ConfusionMatrix
ROCResult
PrecisionRecallResult
CalibrationResult
LearningCurveResult
CrossValidationResult
OptimizationTrace
ProjectionResult
ClusterResult
TreeLayout
```

The separate `TiliaMakieRecipes` package should define recipes for these types.

Do not make plotting part of the core package, including through a package
extension or weak dependency.

---

## 16. Persistence

Do not use Julia `Serialization` as the only long-term model format.

Use a versioned schema-based format.

Suggested structure:

```text
model/
├── manifest.toml
├── specification.toml
├── schema.toml
├── report.toml
└── arrays/
    ├── coefficients.bin
    ├── means.bin
    └── components.bin
```

The manifest should record:

- format version;
- estimator identifier;
- estimator schema version;
- package version;
- Julia version;
- array types;
- array dimensions;
- endianness;
- graph structure;
- category mappings;
- checksums.

Implement explicit migrations:

```julia
migrate(::Val{1}, ::Val{2}, representation)
```

Use `Serialization` only for temporary caches.

---

## 17. Randomness and reproducibility

Use explicit deterministic random streams.

Derive substreams from:

- root seed;
- graph node identity;
- cross-validation fold;
- hyperparameter trial;
- worker identity;
- algorithm stage.

Parallel scheduling order must not affect results when deterministic execution is requested.

Reports should record:

- root seed;
- stream identifiers;
- deterministic mode;
- backend;
- thread count where relevant.

Avoid hidden dependence on global random state.

---

## 18. Numerical policy

Define a central numerical policy.

It should control:

- default floating-point type;
- internal accumulation type;
- tolerance scaling;
- convergence defaults;
- stable summation where needed;
- overflow and underflow handling;
- missing and infinite values;
- deterministic reductions;
- sparse centering policy.

Support at least:

- `Float32`;
- `Float64`.

Do not assume all models are numerically stable under both types. Declare and test support explicitly.

---

## 19. Error handling

Use typed errors for predictable failures.

Examples:

```julia
InvalidHyperparameterError
SchemaMismatchError
UnsupportedDataError
UnsupportedBackendError
ConvergenceError
NumericalFailureError
PersistenceVersionError
GraphValidationError
LeakageError
```

Error messages should include:

- the failing model or node;
- the incompatible input property;
- expected behavior;
- remediation where possible.

Do not hide numerical failures behind `NaN` outputs.

---

## 20. Reports and diagnostics

Every fitted object should expose a structured report.

Possible report fields:

- convergence status;
- objective history;
- iteration count;
- fit duration;
- compilation duration;
- backend;
- device;
- memory estimates;
- data conversion steps;
- warnings;
- feature names;
- class ordering;
- numerical rank;
- regularization;
- fallback operations;
- solver diagnostics.

Reports should be machine-readable Julia values, not only log strings.

Logging should supplement reports, not replace them.

---

## 21. Initial model set

Implement models in an order that exercises the architecture.

## Phase A: foundational supervised learning

- ordinary least squares;
- ridge regression;
- logistic regression;
- binary classification metrics;
- multiclass classification metrics;
- standardization;
- imputation;
- one-hot encoding;
- train/test split;
- k-fold cross-validation.

## Phase B: decomposition and clustering

- PCA;
- truncated SVD;
- k-means;
- Gaussian naive Bayes;
- linear discriminant analysis;
- quadratic discriminant analysis;
- Gaussian mixtures.

## Phase C: neighbors and sparse models

- brute-force nearest neighbors;
- k-nearest-neighbor classifier;
- k-nearest-neighbor regressor;
- sparse logistic regression;
- lasso;
- elastic net;
- coordinate descent.

## Phase D: trees and ensembles

- decision tree classifier;
- decision tree regressor;
- random forest;
- extra trees;
- histogram-based gradient boosting;
- isolation forest.

## Phase E: kernel and neural methods

- kernel ridge regression;
- support vector classifier;
- support vector regressor;
- shallow MLP;
- restricted Boltzmann machine.

Do not add a model family until the infrastructure it depends on is stable.

---

## 22. Standard model layout

Each substantial model family should follow a predictable internal organization.

Example:

```text
src/models/trees/
├── types.jl
├── criterion.jl
├── histogram.jl
├── splitter.jl
├── builder.jl
├── pruning.jl
├── predict.jl
├── regressor.jl
├── classifier.jl
└── report.jl
```

A model implementation should typically contain:

- specification type;
- fitted type;
- validation;
- data preparation;
- fit algorithm;
- prediction;
- reporting;
- persistence schema;
- conformance registration;
- model-specific tests.

Avoid class hierarchies.

Use composition, shared kernels, and dispatch.

---

## 23. Testing requirements

## 23.1 Conformance suite

Every estimator must pass shared tests for:

- fit/predict dimensional consistency;
- deterministic seeded behavior;
- input immutability;
- target validation;
- weight semantics;
- class ordering;
- probability normalization;
- sparse/dense agreement where supported;
- `Float32` and `Float64`;
- serialization round trip;
- graph composition;
- CPU/accelerator agreement where supported;
- convergence reporting;
- degenerate inputs;
- allocation limits during inference.

## 23.2 Mathematical invariants

Test mathematical properties, not only exact outputs.

Examples:

- PCA components are orthonormal.
- Explained variance is nonnegative.
- Probabilities sum to one.
- Ridge solutions satisfy the normal equations within tolerance.
- K-means objective does not increase across iterations.
- Standardized features meet the documented mean and variance convention.
- Tree leaves partition all training observations.
- Cross-validation splits do not overlap incorrectly.
- Sparse and dense paths agree within tolerance.

## 23.3 Reference fixtures

Generate offline reference fixtures using trusted external systems.

Possible references:

- scikit-learn;
- R;
- MATLAB;
- published benchmark datasets;
- symbolic or high-precision calculations.

External systems must not become runtime dependencies.

Store:

- inputs;
- expected outputs;
- tolerance metadata;
- source version;
- generation script.

## 23.4 Regression tests

Every fixed bug should add a regression test.

## 23.5 Allocation tests

Track allocations for:

- inference;
- repeated transforms;
- graph execution;
- metrics;
- hot kernels.

Do not require zero allocation everywhere. Set realistic budgets.

---

## 24. Benchmarking

Maintain separate benchmark suites for:

- kernels;
- fit time;
- inference time;
- memory usage;
- compilation latency;
- graph optimization;
- CPU scaling;
- accelerator scaling;
- sparse workloads;
- tabular preprocessing.

Benchmarks should include small, medium, and large problem sizes.

Always separate:

- compilation time;
- first-call latency;
- steady-state runtime.

Do not claim speedups using only end-to-end numbers that obscure compilation and transfer costs.

---

## 25. Documentation

Organize documentation around workflows before algorithms.

Primary workflow:

```text
load data
→ inspect schema
→ split
→ build pipeline
→ fit
→ evaluate
→ tune
→ inspect
→ save
```

Documentation sections:

1. Getting started.
2. Data and schemas.
3. Pipelines and graphs.
4. Models.
5. Metrics.
6. Model selection.
7. Acceleration.
8. Differentiation.
9. Persistence.
10. Numerical behavior.
11. Extending the project.
12. Internals.

Every public type and function must have a docstring and at least one example.

---

## 26. Coding standards

### 26.1 Style

- Use explicit, descriptive names.
- Prefer small functions.
- Avoid global mutable state.
- Avoid hidden allocations in hot paths.
- Use type parameters only when they materially improve dispatch or performance.
- Avoid encoding large dynamic configurations in types.
- Keep error messages actionable.
- Keep public behavior stable.

### 26.2 Performance

- Write generic Julia first.
- Measure before specializing.
- Use `@inbounds` only after correctness tests.
- Use `@simd` only when safe.
- Use mutation internally where justified.
- Avoid unnecessary materialization.
- Preserve sparse structure.
- Separate setup from hot loops.
- Benchmark with realistic data shapes.

### 26.3 Type stability

Public hot paths should be type-stable.

Use inference tools during development.

Do not require every diagnostic or configuration object to be fully type-specialized if that increases compile time without runtime benefit.

### 26.4 Numerical clarity

For each algorithm, document:

- objective function;
- optimization method;
- convergence criterion;
- regularization convention;
- intercept convention;
- class ordering;
- variance convention;
- precision assumptions.

---

## 27. Continuous integration

CI should include:

- supported Julia versions;
- Linux;
- macOS;
- Windows;
- `Float32` and `Float64` tests;
- sparse tests;
- thread-count variations;
- documentation build;
- formatting and static checks;
- package quality checks;
- optional Reactant tests;
- separate `TiliaMakieRecipes` package tests;
- persistence compatibility tests.

Use separate jobs for optional integrations so core CI remains lightweight.

Add nightly jobs for:

- larger numerical tests;
- benchmark trend detection;
- accelerator tests;
- randomized property tests.

---

## 28. Contribution workflow

Every new model or major feature should include:

1. design note;
2. public API proposal;
3. mathematical specification;
4. implementation;
5. conformance tests;
6. invariant tests;
7. reference tests;
8. documentation;
9. benchmark;
10. persistence support;
11. report definition;
12. backend capability declaration.

Avoid merging algorithms that bypass shared infrastructure without a documented reason.

---

## 29. Initial implementation milestones

## Milestone 0: repository bootstrap

Deliver:

- package skeleton;
- CI;
- documentation;
- style guide;
- dependency policy;
- contribution guide;
- public API placeholder;
- test environments;
- benchmark environments.

Exit condition:

```julia
using Tilia
```

loads with only hard dependencies.

## Milestone 1: core semantics

Deliver:

- model specification interface;
- fitted-object interface;
- traits;
- fit context;
- schema types;
- dataset types;
- reports;
- typed errors;
- conformance-test framework.

Exit condition:

A trivial mean regressor can fit, predict, report, and serialize.

## Milestone 2: semantic graph

Deliver:

- graph nodes;
- ports;
- graph builder;
- topological validation;
- schema propagation;
- fit/inference distinction;
- leakage validation;
- CPU graph interpreter.

Exit condition:

A pipeline of column selection, standardization, and mean regression runs through the graph.

## Milestone 3: numerical kernels

Deliver:

- reductions;
- stable log/exp functions;
- normalization;
- weighted statistics;
- pairwise Euclidean distance;
- core losses;
- sparse scaling.

Exit condition:

Kernels have correctness, type, allocation, and benchmark coverage.

## Milestone 4: first complete ML workflow

Deliver:

- table ingestion;
- categorical columns;
- imputation;
- one-hot encoding;
- standardization;
- ordinary least squares;
- ridge regression;
- logistic regression;
- train/test split;
- k-fold cross-validation;
- accuracy;
- log loss;
- RMSE;
- persistent model format.

Exit condition:

A mixed-type table can be trained, cross-validated, inspected, saved, loaded, and used for prediction.

## Milestone 5: graph optimization

Deliver:

- dead-node elimination;
- transform fusion;
- conversion elimination;
- buffer planning;
- execution tracing;
- graph visualization data.

Exit condition:

Optimized and unoptimized graphs produce equivalent results.

## Milestone 6: Reactant prototype

Deliver:

- optional backend;
- lowering for core dense kernels;
- accelerated standardization;
- accelerated matrix multiplication;
- accelerated logistic-regression objective;
- compilation cache;
- fallback diagnostics.

Exit condition:

A dense standardization-plus-logistic-regression pipeline can run through Reactant with explicit device reporting.

## Milestone 7: model breadth

Deliver:

- PCA;
- k-means;
- naive Bayes;
- discriminant analysis;
- Gaussian mixtures;
- nearest neighbors.

Exit condition:

The architecture supports decomposition, iterative clustering, probabilistic models, and pairwise methods without new core abstractions.

## Milestone 8: irregular algorithms

Deliver:

- decision trees;
- random forests;
- extra trees;
- histogram gradient boosting.

Exit condition:

The system supports branch-heavy CPU algorithms while preserving the same user API and reports.

---

## 30. First coding tasks

Create the following files first:

```text
src/Tilia.jl
src/core/api.jl
src/core/estimator.jl
src/core/fitted.jl
src/core/traits.jl
src/core/context.jl
src/core/reports.jl
src/core/errors.jl
src/data/schema.jl
src/data/dataset.jl
src/graph/node.jl
src/graph/graph.jl
src/graph/builder.jl
src/graph/validation.jl
src/graph/interpreter.jl
test/runtests.jl
test/core/
test/data/
test/graph/
test/conformance/
```

Implement a minimal end-to-end path:

```julia
model = MeanRegressor()
fitted = fit(model, X, y)
ŷ = predict(fitted, Xnew)
r = report(fitted)
```

Then implement:

```julia
pipeline = Chain(
    Standardize(),
    MeanRegressor(),
)
```

The first pipeline should execute through the semantic graph and CPU interpreter.

Do not begin logistic regression, PCA, or Reactant integration until this path is complete.

---

## 31. Minimum API sketch

The initial API can be approximately:

```julia
abstract type AbstractEstimator end
abstract type AbstractFittedEstimator end
abstract type AbstractTransformer <: AbstractEstimator end
abstract type AbstractPredictor <: AbstractEstimator end

fit(model::AbstractEstimator, args...; context=default_context())
predict(model::AbstractFittedEstimator, X)
transform(model::AbstractFittedEstimator, X)
report(model::AbstractFittedEstimator)

capabilities(::Type{<:AbstractEstimator})
input_contract(::AbstractEstimator)
output_schema(::AbstractEstimator, input_schema)
```

Avoid finalizing these names before implementing the first complete workflow.

The API should be validated against actual usage rather than abstract speculation.

---

## 32. Architectural review checkpoints

Pause and review the architecture after:

- the first fitted estimator;
- the first transformer pipeline;
- the first table workflow;
- the first sparse model;
- the first graph optimization;
- the first Reactant backend;
- the first tree model.

At each checkpoint, evaluate:

- whether abstractions are still minimal;
- whether type complexity is growing;
- whether error messages remain understandable;
- whether compile time is acceptable;
- whether backend logic is leaking into models;
- whether reports are sufficient;
- whether persistence remains stable;
- whether new models require inappropriate special cases.

Refactor before adding breadth.

---

## 33. Project success criteria

The project should be considered successful when it provides:

- a coherent classical ML workflow;
- minimal hard dependencies;
- native implementations;
- consistent model semantics;
- safe graph-based pipelines;
- reproducible evaluation;
- structured diagnostics;
- stable persistence;
- strong CPU performance;
- selective accelerator execution;
- a contribution model that does not fragment the stack.

Raw algorithm count is not the primary success metric.

The most important deliverable is a stable set of contracts that allows the package to grow for years without becoming a collection of unrelated implementations.
