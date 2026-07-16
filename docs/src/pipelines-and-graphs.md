# Pipelines and graphs

Tilia composition is both a user-facing pipeline API and a semantic graph
representation. Transformations are fitted only on training data, while graph
structure remains available for validation, reporting, tracing, optimization,
and backend placement.

## Sequential composition with `Chain`

`Chain` sends the output of each step to the next:

```julia
model = Chain(
    Standardize(),
    PCA(n_components=3),
    LogisticRegression(),
)

fitted = fit(model, Xtrain, ytrain)
predictions = predict(fitted, Xtest)
```

During cross-validation, the entire chain is refitted inside each fold. Means,
scales, components, levels, and model parameters therefore never learn from
held-out rows.

The last step determines the graph operation: a predictor supports `predict`
and possibly `predict_proba`; a transformer supports `transform`.

## Column-aware preprocessing

`ColumnMap` applies a transformer to named or indexed columns and concatenates
the resulting numeric branches in mapping order:

```julia
preprocess = ColumnMap(
    (:age, :income) => Chain(Impute(), RobustScale()),
    :color => Chain(
        Impute(),
        OneHotEncode(passthrough_numeric=false),
    ),
)

model = Chain(preprocess, LogisticRegression())
fitted = fit(model, training_table, labels)
```

Names are accepted for `ColumnTable` or other Tables.jl inputs. Matrix inputs
use integer indices. Branch outputs must be numeric before concatenation.

## Structural combinators

The five composition primitives have distinct roles:

| Primitive | Meaning |
|:--|:--|
| `Chain(a, b, ...)` | Sequential fit and execution |
| `Select(columns)` | Stable column selection |
| `ColumnMap(key => transformer, ...)` | Select, transform, and concatenate heterogeneous columns |
| `Parallel(a, b, ...)` | Apply several transformations to the same input and return branch outputs |
| `Concatenate()` | Combine a tuple of numeric branch outputs column-wise |

Explicit branching can be written with `Parallel` and `Concatenate`:

```julia
features = Chain(
    Parallel(
        Chain(Select(1:2), Standardize()),
        Chain(Select(3:4), PolynomialFeatures(degree=2)),
    ),
    Concatenate(),
)
```

`ColumnMap` is usually more concise for heterogeneous tables; explicit
parallel composition is useful when branches reuse the same input or express
more general graph structure.

## From composition to a semantic graph

Fitting a chain builds nodes for selections, transformations, conversions,
branches, concatenation, and prediction. The CPU interpreter validates:

- acyclic topology and reachable outputs;
- feature counts and semantic input/output schemas;
- supported estimator operations and weight flow;
- training-only fitting of every learned transformation;
- backend compatibility and explicit transfer boundaries.

The resulting `FittedGraph` retains fitted node state and a graph-level report:

```julia
fitted = fit(model, X, y)
summary = report(fitted)
summary.details
```

## Optimization and execution planning

Graph optimization is semantics-preserving. Available passes include constant
folding, dead-node elimination, redundant-conversion removal, compatible
transform fusion, device placement, transfer coalescing, and buffer planning.

These passes operate on declared node semantics rather than pattern-matching an
opaque user function. An execution plan can therefore account for lifetimes,
reuse, placement, and transfers while reports retain the decisions made.

Optional Reactant execution currently supports a deliberately limited subset;
see [Acceleration](acceleration.md). Unsupported placement raises an error or
uses an explicitly requested fallback policy—it does not silently claim that
the entire graph ran on an accelerator.

## Tracing and diagnostics

Tracing exposes node operations and execution behavior for inspection. Fit
reports record observations, feature counts, backend, warnings, numerical
policy, deterministic stream metadata, and step-specific details such as
timings or objective history.

For model diagnostics and graph outputs, the separate Makie package provides
plots without adding visualization dependencies to core Tilia; see
[Visualization with Makie](visualization.md).

## Designing reliable pipelines

1. Put every learned preprocessing operation inside the fitted graph.
2. Use `ColumnMap` when different columns need different missingness,
   categorical, or scaling behavior.
3. Use `Parallel` and `Concatenate` when the branch structure itself is part of
   the model.
4. Inspect `input_contract`, `output_schema`, and `capabilities` when composing
   unfamiliar estimators.
5. Evaluate the complete model specification, not preprocessed data produced
   before the split.
