# Tilia.jl

Tilia is an experimental, Julia-native classical machine-learning stack. It
brings estimators, preprocessing, semantic computation graphs, schemas,
evaluation, persistence, diagnostics, and optional acceleration under one
explicit set of contracts.

```julia
using Tilia

X = column_table((
    age    = [22, 25, 31, 36, 42, 47, 53, 58],
    income = [28, 35, 46, 52, 61, 73, 82, 94] .* 1_000.0,
    plan   = [:basic, :basic, :plus, :basic, :plus, :pro, :plus, :pro],
))
y = [:stay, :leave, :stay, :leave, :stay, :stay, :leave, :stay]

model = Chain(
    ColumnMap(
        (:age, :income) => Standardize(),
        :plan => OneHotEncode(passthrough_numeric=false),
    ),
    LogisticRegression(lambda=0.1),
)

Xtrain, Xtest, ytrain, ytest, _, _ =
    train_test_split(X, y; test_size=0.25, stratify=y, seed=42)

fitted = fit(model, Xtrain, ytrain; context=FitContext(seed=42))
predictions = predict(fitted, Xtest)
probabilities = predict_proba(fitted, Xtest)

accuracy_score(ytest, predictions)
confusion_matrix(ytest, predictions)
cross_validate(model, X, y; cv=KFold(4; shuffle=true, seed=42)).scores
report(fitted)                     # schema, graph, numerics, timings, warnings

save_model("customer_model.tilia", fitted)
restored = load_model("customer_model.tilia")
predict(restored, Xtest) == predictions
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
| Models | Linear and sparse models, decomposition, clustering, mixtures, neighbors, probabilistic classifiers, trees, ensembles, kernels, shallow neural models |
| Composition | `Chain`, `ColumnMap`, `Select`, `Parallel`, `Concatenate`, semantic graph validation and optimization |
| Evaluation | Metrics, deterministic splits, cross-validation, parameter tuning, permutation importance, structured diagnostic results |
| Operations | Numerical policies, deterministic random streams, fit reports, structural persistence, tracing, optional Reactant execution |
| Visualization | A separate `TiliaMakieRecipes` package with diagnostic, explanatory, dimensionality-reduction, clustering, and `Axis3` plots |

Use [`model_catalog`](models.md#Finding-a-model-programmatically) and
`capabilities(model)` to discover machine-readable support instead of assuming
that every estimator accepts sparse input, missing values, weights,
probabilities, or incremental fitting.

## Why Tilia?

Tilia is being developed for users who want the whole classical-ML workflow to
remain inspectable inside Julia:

1. **Specifications and learned state are different values.** Reusing a model
   specification cannot silently carry parameters learned by an earlier fit.
2. **Pipelines are semantic graphs.** Column selection, branch structure,
   conversions, and predictions remain visible to validation, tracing,
   optimization, persistence, and future backends.
3. **Data contracts travel with fitted objects.** Schemas record feature order,
   logical and physical types, missingness rules, categorical levels, target
   metadata, and generated-column provenance.
4. **Numerical behavior is an API concern.** A `FitContext` carries numerical
   policy, backend choice, deterministic seed streams, compilation cache, and
   fallback policy; reports record what actually happened.
5. **Optional systems stay optional.** Reactant execution, differentiation, and
   Makie visualization do not enlarge the core dependency surface for users
   who do not need them.
6. **Reference behavior is testable.** The repository includes conformance,
   numerical-reference, persistence, graph, allocation, and visualization
   tests rather than treating estimator output as an opaque implementation
   detail.

Tilia is not presented as a drop-in replacement for a mature ecosystem, and
its experimental status matters. Its purpose is to explore a coherent,
Julia-native stack in which model semantics, execution, and diagnostics can be
developed together.

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
