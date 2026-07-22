# Model selection

Tilia's selection API keeps row ownership explicit and refits complete model
specifications inside every fold. Preprocessing outside the evaluated pipeline
is not automatically protected from leakage; put learned transforms in a
`Chain` or semantic graph.

## Train/test splitting

```julia
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y;
    test_size=0.2,
    shuffle=true,
    seed=42,
    stratify=y,
)
```

`test_size` may be an observation count or a fraction. To also receive the
sorted, disjoint row indices, pass `return_indices=true`; the result then has
the form `(Xtrain, Xtest, ytrain, ytest, train_indices, test_indices)`.
Stratification requires at least two observations per class and preserves each
class in both partitions when the requested size permits it.

## K-fold definitions

```julia
cv = KFold(5; shuffle=true, seed=42)
folds = split(cv, length(y))
```

Fold sizes differ by at most one. Each tuple contains training and test indices;
every observation appears in exactly one test fold. Shuffling uses the
splitter's local seed rather than Julia's global RNG.

## Leakage-safe cross-validation

```julia
model = Chain(Standardize(), RidgeRegression(lambda=0.1))
evaluation = cross_validate(model, X, y; cv=cv)
```

`evaluate` is an alias for the same workflow. `CrossValidationResult` contains:

| Field | Meaning |
|:--|:--|
| `scores` | One score per fold |
| `fitted_models` | Independently fitted fold models |
| `fold_reports` | Report from each fit |
| `train_indices`, `test_indices` | Exact rows used by every fold |

Classification defaults to accuracy; regression defaults to RMSE. Supply
`scoring(truth, prediction)` to change the fold statistic:

```julia
evaluation = cross_validate(
    model, X, y;
    cv=cv,
    scoring=(truth, prediction) -> mean_squared_error(truth, prediction),
    context=FitContext(seed=42),
)
```

The context derives a named stream for every fold, so stochastic models remain
reproducible without sharing mutable RNG progress.

## Exhaustive parameter grids

`tune` evaluates the Cartesian product of a named parameter grid:

```julia
search = tune(
    ElasticNet(), X, y;
    parameter_grid=(
        lambda=[0.01, 0.1, 1.0],
        l1_ratio=[0.2, 0.5, 0.8],
    ),
    cv=cv,
    context=FitContext(seed=42),
)
```

Grid names must be constructor fields of the estimator being tuned. Each value
must be a vector or tuple. `TuningResult` contains the best model specification,
parameters, mean score, all trials and fold scores, and—by default—a model
refitted on the complete input.

Classification scores are maximized and regression scores minimized by
default. For custom scoring, state the direction explicitly when necessary:

```julia
search = tune(
    RidgeRegression(), X, y;
    parameter_grid=(lambda=10.0 .^ (-3:1),),
    scoring=(truth, prediction) -> -root_mean_squared_error(truth, prediction),
    maximize=true,
    refit=false,
)
```

## Interpreting results

Do not select a model from its test-set performance. Use training folds for
selection, reserve a final test set for the chosen workflow, and inspect score
dispersion and fold reports rather than only the mean.

The separate Makie package provides fold plots, learning and validation curves,
tuning heatmaps, parallel-coordinate views, three-dimensional score
landscapes, and multi-model comparisons. See
[Visualization with Makie](visualization.md).
