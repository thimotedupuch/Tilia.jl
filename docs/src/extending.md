# Extending Tilia

An estimator extension is more than a `fit` method. It joins the capability,
schema, reporting, graph, numerical, persistence, testing, and documentation
contracts used by generic Tilia workflows.

## Minimal estimator shape

Define an immutable specification and a separate fitted type:

```julia
struct MyRegressor <: Tilia.AbstractPredictor
    offset::Float64
end

struct FittedMyRegressor{M,T,R,S} <: Tilia.AbstractFittedEstimator
    model::M
    prediction::T
    report::R
    schema::S
end

Tilia.capabilities(::Type{<:MyRegressor}) = (
    task=:regression,
    sparse=false,
    missing=false,
    weights=false,
    partial_fit=false,
    probabilistic=false,
)
```

The specification contains only hyperparameters. Learned values, input schema,
and report belong to the fitted object.

## Fit, inference, and reporting

An implementation should validate dimensions and declared capabilities before
computing:

```julia
function Tilia.fit(model::MyRegressor, X::AbstractMatrix, y::AbstractVector;
                   weights=nothing, context=Tilia.default_context())
    weights === nothing || throw(ArgumentError("MyRegressor does not support weights"))
    size(X,1) == length(y) || throw(DimensionMismatch("X and y must agree"))
    prediction = sum(y) / length(y) + model.offset
    schema = Tilia.infer_schema(X)
    fitted_report = Tilia.FitReport(
        observations=size(X,1), features=size(X,2), context=context,
        details=(offset=model.offset,),
    )
    FittedMyRegressor(model, prediction, fitted_report, schema)
end

Tilia.predict(fitted::FittedMyRegressor, X::AbstractMatrix) =
    fill(fitted.prediction, size(X,1))
Tilia.report(fitted::FittedMyRegressor) = fitted.report
```

Some helpers in this example are internal rather than exported. External
packages should qualify them and accept that internal APIs may evolve until a
dedicated extension surface is stabilized.

## Schema and graph integration

For transformers, implement `transform` and an accurate
`output_schema(model, input_schema)`. Declare whether feature count is
preserved and whether sparse or missing inputs are genuinely supported.

Graph nodes derive behavior from capabilities and contracts. Incorrect traits
are bugs: claiming sparse, missing, probabilistic, weighted, or incremental
support causes generic workflows to rely on that promise.

## Numerical implementation

Reuse `Tilia.Kernels` and `Tilia.Solvers` for shared primitives. Respect the
supplied `FitContext`, derive named subcontexts for stochastic suboperations,
and use the effective numerical policy rather than defining unrelated global
tolerances.

Reports should capture convergence, iterations, objective history, warnings,
backend, and algorithm-specific details needed to diagnose a fit.

## Persistence is explicit

External fitted types are not automatically accepted by Tilia's structural
persistence registry. A persistable estimator needs an intentional format
contract, versioning decision, migration behavior, round-trip fixtures, and
corruption tests. Do not fall back to Julia `Serialization` inside a Tilia
artifact.

## Definition of done

A production-quality estimator contribution should include:

1. immutable specification and fitted state;
2. complete capability declarations and input validation;
3. fit, inference or transform, and report methods;
4. schema propagation and graph-composition tests;
5. Float32/Float64, allocation, sparse, missing, and weight conformance where declared;
6. deterministic reference fixtures from an independent implementation;
7. typed failure tests and edge cases;
8. persistence support when in scope;
9. benchmarks separating compilation, fitting, inference, and allocation;
10. constructor docstrings and updates to model and numerical-contract documentation.

Inspect existing estimators of the nearest family before choosing conventions;
consistency is part of the public API.
