# Tasks that benefit from Tilia's execution model

These examples focus on work that becomes straightforward when preprocessing,
models, schemas, execution plans, fitted state, and reports belong to one
system. They are not intended as one-to-one API comparisons.

## Fit heterogeneous preprocessing without losing column meaning

Map transformations to named columns, then fit the resulting semantic graph.
Each learned preprocessing state belongs to the fitted graph and is reused at
inference.

```julia
using Tilia

table = (
    age=[20.0, 24.0, 41.0, 53.0],
    income=[28_000.0, 35_000.0, 72_000.0, 91_000.0],
    city=[:Lyon, :Paris, :Lyon, :Paris],
)
labels = [:basic, :basic, :premium, :premium]

features = ColumnMap(
    (:age, :income) => RobustScale(),
    :city => OneHotEncode(passthrough_numeric=false,
                          handle_unknown=:ignore),
)
model = Chain(features, LogisticRegression(lambda=0.2))
fitted = fit(model, table, labels)

predict_proba(fitted, (age=[37.0], income=[60_000.0], city=[:Grenoble]))
report(fitted).details.propagated_schemas
```

## Build a multi-view feature graph

Run several learned representations over the same input, concatenate their
outputs, and fit one predictor. Branch dimensions and fitted states remain
inspectable.

```julia
using Tilia, Random

rng = Xoshiro(12)
X = randn(rng, 200, 8)
y = 2 .* X[:, 1] .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 200)

model = Chain(
    Parallel(
        Standardize(),
        PCA(n_components=3),
        PolynomialFeatures(degree=2, include_bias=false,
                           interaction_only=true),
    ),
    Concatenate(),
    RidgeRegression(lambda=0.5),
)

fitted = fit(model, X, y)
predict(fitted, X[1:5, :])
report(fitted.fitted_nodes[1]).details.output_widths
```

## Ask what numerical work a fit actually performed

The fitted report contains lowered fit and inference graphs. This makes solver,
shape, representation, and device choices data rather than conventions hidden
inside an estimator.

```julia
using Tilia, Random

X = randn(500, 12)
y = X[:, 1] .- 0.4 .* X[:, 2]
fitted = fit(Chain(Standardize(), LinearRegression()), X, y)

fit_graph = report(fitted).details.fit_execution_graph
inference_graph = report(fitted).details.inference_execution_graph

[(primitive.operation, primitive.device) for primitive in fit_graph.primitives]
[(node.operation, node.input_shape, node.output_shape)
 for node in inference_graph.nodes]

# The linear solve is lowered to standard-library QR primitives.
any(primitive.operation == :qr_factorization for primitive in fit_graph.primitives)
```

## Optimize a fitted graph and verify semantic equivalence

Optimization acts on fitted state, preserves predictions, and records what was
changed. Here two affine transforms can be fused for inference.

```julia
using Tilia, Random

X = randn(300, 6)
y = X[:, 1] .+ 0.2 .* X[:, 2]

model = Chain(
    Standardize(center=true, scale=false),
    Standardize(center=false, scale=true),
    LinearRegression(),
)

fitted = fit(model, X, y)
optimized = Tilia.optimize(fitted)

@assert predict(optimized, X) ≈ predict(fitted, X)
report(optimized).details.optimization
Tilia.graph_data(optimized)
```

## Trace a real inference call node by node

Tracing returns the prediction together with per-node latency, shapes, and
materialized output sizes. It uses the same fitted graph that serves normal
predictions.

```julia
using Tilia, Random

X = randn(1_000, 20)
y = ifelse.(X[:, 1] .+ X[:, 2] .> 0, :yes, :no)
fitted = fit(Chain(Standardize(), LogisticRegression()), X, y)

execution = Tilia.trace(fitted, X[1:100, :]; operation=:predict_proba)
execution.output
[(node.operation, node.nanoseconds, node.input_shape,
  node.output_shape, node.output_bytes) for node in execution.nodes]
```

## Reproduce stochastic work through named substreams

A root context deterministically derives streams for graph nodes, restarts,
folds, tree iterations, feature permutations, and other nested work. Adding
unrelated random work does not require passing mutable RNGs through every API.

```julia
using Tilia, Random

X = randn(400, 5)
context = FitContext(seed=2026)

first = fit(KMeans(n_clusters=4, n_init=5), X; context=context)
second = fit(KMeans(n_clusters=4, n_init=5), X;
             context=FitContext(seed=2026))

@assert first.centers == second.centers
@assert report(first).root_seed == report(second).root_seed

importance = permutation_importance(
    fit(RidgeRegression(), X, X[:, 1]), X, X[:, 1];
    n_repeats=5,
    context=derive_context(context, :inspection),
)
importance.mean_importance
```

## Persist the complete executable model artifact

Persistence includes learned graph nodes, schemas, reports, typed arrays,
checksums, and format metadata. Loading does not depend on Julia's opaque
`Serialization` format.

```julia
using Tilia, Random

X = randn(100, 4)
y = ifelse.(X[:, 1] .> 0, :positive, :negative)
fitted = fit(Chain(Standardize(), LogisticRegression()), X, y)

save_model("classifier-artifact", fitted)
restored = load_model("classifier-artifact")

@assert predict(restored, X) == predict(fitted, X)
@assert report(restored).details.loaded
```
