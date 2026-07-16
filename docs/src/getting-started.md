# Getting started

This page follows one workflow from data preparation through evaluation and
persistence. Tilia uses the same small API for a standalone model and a
composed pipeline.

## Install and load

From a checkout of the repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then load Tilia:

```julia
using Tilia
```

Makie plotting and optional integrations are separate packages or weak
dependencies; they are not required for this workflow.

## Prepare observations and targets

Rows are observations and columns are features:

```julia
X = [
    1.0  0.2
    1.4  0.4
    2.8  1.1
    3.2  1.5
    4.1  2.0
    4.5  2.4
]
y = [:small, :small, :small, :large, :large, :large]
```

For heterogeneous data, pass any Tables.jl-compatible source or create a
`Dataset`; see [Data and schemas](data-and-schemas.md). The direct matrix API
is useful when every feature is already finite and numeric.

## Declare, fit, and predict

A model specification contains hyperparameters, not learned state:

```julia
model = Chain(
    Standardize(),
    LogisticRegression(lambda=0.1, max_iterations=100),
)

fitted = fit(model, X, y)
labels = predict(fitted, X)
probabilities = predict_proba(fitted, X)
```

`model` remains unchanged. `fitted` stores the fitted standardization, learned
classifier parameters, schema, graph execution information, and reports.
Probability columns follow the fitted class order.

```julia
report(fitted)
```

Unsupported operations fail explicitly. For example, calling
`predict_proba` on a non-probabilistic estimator raises an error instead of
inventing a probability interpretation.

## Hold out data

Use a deterministic split before fitting when estimating generalization:

```julia
Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(
    X, y; test_size=0.33, shuffle=true, seed=42, stratify=y,
)

trained = fit(model, Xtrain, ytrain)
test_predictions = predict(trained, Xtest)
accuracy_score(ytest, test_predictions)
```

Split helpers return indices, which keeps ownership and row selection
explicit. Pipelines fit every preprocessing step using only the training rows.

## Cross-validation and tuning

Cross-validation refits the complete pipeline in each fold:

```julia
cv = KFold(3; shuffle=true, seed=42)
evaluation = cross_validate(model, X, y; cv=cv)
evaluation.scores
```

Search model parameters with a named grid:

```julia
tuned = tune(
    LogisticRegression(), X, y;
    parameter_grid=(lambda=[0.01, 0.1, 1.0],),
    cv=cv,
)

tuned.best_parameters
tuned.best_score
tuned.fitted_model
```

Classification scores are maximized by default and regression scores are
minimized. Custom scoring functions can override that behavior with
`maximize`.

## Inspect and persist

Capabilities describe supported operations:

```julia
capabilities(LogisticRegression())
input_contract(model)
```

Persist a fitted object using Tilia's structural format:

```julia
save_model("classifier.tilia", trained)
restored = load_model("classifier.tilia")
predict(restored, Xtest) == test_predictions
```

See [Persistence](persistence.md) for format guarantees and migration behavior.

## Add visualization when needed

The separate plotting package consumes Tilia's fitted objects and semantic
results without making Makie a dependency of Tilia:

```julia
using TiliaMakieRecipes, CairoMakie

plot(evaluation)
decisionboundaryplot(trained, X, y)
```

See [Visualization with Makie](visualization.md) for recipes and the full
plotting API.

## The core pattern

Most Tilia workflows reduce to the same sequence:

```text
declare model → fit on training data → inspect report
              → predict/transform → evaluate → persist or visualize
```

The following pages expand each part without changing this separation between
specification, fitted state, and semantic results.
