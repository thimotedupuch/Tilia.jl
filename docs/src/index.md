# Tilia.jl

Tilia is an experimental, Julia-native classical machine-learning stack. It
brings estimators, preprocessing, semantic computation graphs, schemas,
evaluation, persistence, diagnostics, and optional acceleration under one
explicit set of contracts.

```julia
using Tilia

# Rows are homes; columns are floor area, bedrooms, and building age.
X = [
     52.0  1  38
     61.0  2  25
     70.0  2  18
     78.0  3  30
     85.0  3  12
     93.0  3   8
    105.0  4  20
    116.0  4   6
    128.0  4  15
    142.0  5   4
    155.0  5  10
    170.0  5   2
]
price = [162.0, 188.0, 214.0, 226.0, 264.0, 291.0,
         305.0, 348.0, 367.0, 412.0, 438.0, 486.0] # thousands

Xtrain, Xtest, ytrain, ytest, _, _ =
    train_test_split(X, price; test_size=0.25, seed=42)

model = Chain(
    Standardize(),
    RidgeRegression(lambda=0.2),
)

fitted = fit(model, Xtrain, ytrain)
predictions = predict(fitted, Xtest)

root_mean_squared_error(ytest, predictions)
report(fitted)
```

Observations occupy rows and features occupy columns. Model values such as
`LogisticRegression(lambda=0.1)` are immutable specifications; `fit` returns a
separate object containing learned state, the training schema, and a structured
report. This separation is used consistently by preprocessing, predictors,
clustering, decomposition, pipelines, persistence, and inspection tools.

## What is included?

Tilia currently provides:

| Area | Facilities |
|:--|:--|
| Data | Matrices, Tables.jl inputs, owned `ColumnTable` storage, `Dataset`, semantic schemas, categorical metadata, weights |
| Preprocessing | Imputation, categorical encoding, standard, min--max and robust scaling, normalization, polynomial features |
| Models | Linear, generalized-linear, robust, ordinal, multi-output and meta-estimators, plus decomposition, clustering, mixtures, neighbors, trees, kernels, and shallow neural models |
| Composition | `Chain`, `ColumnMap`, `Select`, `Parallel`, `Concatenate`, semantic graph validation and optimization |
| Evaluation | Metrics, deterministic splits, cross-validation, parameter tuning, permutation importance, structured diagnostic results |
| Operations | Numerical policies, deterministic random streams, fit reports, structural persistence, tracing, optional Reactant execution |
| Visualization | A separate `TiliaMakieRecipes` package with diagnostic, explanatory, dimensionality-reduction, clustering, and `Axis3` plots |

Use [`model_catalog`](models.md#Finding-a-model-programmatically) and
`capabilities(model)` to discover machine-readable support instead of assuming
that every estimator accepts sparse input, missing values, weights,
probabilities, or incremental fitting.

## Why Tilia?

Tilia is for classical-ML workflows where understanding what was fitted matters
as much as obtaining a prediction. Models are immutable specifications; learned
parameters, training schema, and diagnostics live in the value returned by
`fit`:

```julia
model = RidgeRegression(lambda=0.2)
fitted = fit(model, Xtrain, ytrain)

model.lambda          # configuration
fitted.coefficients   # learned state
report(fitted)        # convergence and execution details
```

Preprocessing and prediction form one leakage-safe workflow. Tilia represents
that workflow as a semantic graph, so its structure remains available for
validation, optimization, tracing, persistence, and optional execution
backends:

```julia
workflow = Chain(
    Standardize(),
    PCA(n_components=2),
    RidgeRegression(lambda=0.1),
)

fitted_workflow = fit(workflow, Xtrain, ytrain)
predict(fitted_workflow, Xtest)
```

Support is explicit rather than discovered through a failed fit. Generic code
can inspect whether a model accepts sparse data, weights, missing values,
probabilistic prediction, or incremental updates:

```julia
capabilities(LogisticRegression())
model_catalog(task=:classification, probabilistic=true)
```

Randomness and numerical choices are also explicit inputs. The same context can
be used to reproduce stochastic work without depending on global RNG state:

```julia
context = FitContext(seed=42)
clusters = fit(KMeans(n_clusters=4), X; context=context)
report(clusters)
```

Finally, fitted graphs can be stored structurally and loaded through the same
public API:

```julia
save_model("house-price-model.tilia", fitted)
restored = load_model("house-price-model.tilia")
predict(restored, Xtest)
```

The core package stays Julia-native and dependency-light. Reactant execution,
automatic differentiation, and Makie visualization are optional integrations.
Tilia is still experimental, but its schemas, numerical contracts, reference
fixtures, conformance tests, and reports are designed to make that maturity
visible rather than implicit.

## Existing ML software

Tilia exists alongside established projects and learns from their different
scopes:

- [scikit-learn](https://scikit-learn.org/stable/) offers a mature Python API
  and broad collection of classical supervised and unsupervised algorithms.
- [MLJ](https://juliaai.github.io/MLJ.jl/dev/) provides a model-agnostic Julia
  interface, composition tools, evaluation, and access to models implemented
  across many packages and languages. Its design is described in the
  [MLJ paper](https://arxiv.org/abs/2007.12285).
- [Flux](https://fluxml.ai/Flux.jl/) and
  [Lux](https://lux.csail.mit.edu/stable/) focus on differentiable programming
  and neural-network construction in Julia.
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) is a specialized,
  production-oriented gradient-boosted-tree system with bindings for several
  languages.
- The [SciML ecosystem](https://sciml.ai/) develops composable scientific
  machine learning around differential equations, simulation, optimization,
  and differentiable scientific programs.

These projects are complementary rather than interchangeable. Choose based on
model coverage, ecosystem integration, deployment needs, maturity, and the
abstractions you want to make explicit. Tilia's distinctive emphasis is a
single native implementation stack for classical models, semantic graph
execution, schemas, numerical contracts, and structured reporting.

## Where to go next

- Follow [Getting started](getting-started.md) for a complete train/evaluate/save workflow.
- Read [Data and schemas](data-and-schemas.md) before working with heterogeneous tables.
- Use [Pipelines and graphs](pipelines-and-graphs.md) for leakage-safe preprocessing and branching.
- Browse [Models](models.md) and the detailed [numerical contracts](model-semantics.md).
- Evaluate with [Metrics](metrics.md) and [Model selection](model-selection.md).
- Add optional [Makie visualization](visualization.md), [acceleration](acceleration.md), or [differentiation](differentiation.md).
- Consult [Persistence](persistence.md), [Extending Tilia](extending.md), and [Internals](internals.md) for operational and development details.
