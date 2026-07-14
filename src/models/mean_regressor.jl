"""Baseline regressor that predicts the (optionally weighted) target mean."""
struct MeanRegressor <: AbstractPredictor end

struct FittedMeanRegressor{M,T,R,S} <: AbstractFittedEstimator
    model::M
    mean::T
    report::R
    schema::S
end

capabilities(::Type{<:MeanRegressor}) = (
    task=:regression, sparse=true, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::MeanRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "MeanRegressor fitting")
    _validate_numeric_matrix(X, "MeanRegressor")
    n = size(X, 1)
    length(y) == n || throw(SchemaMismatchError(
        "MeanRegressor target has length $(length(y)); expected $n observations."))
    n > 0 || throw(UnsupportedDataError("MeanRegressor requires at least one observation."))
    all(isfinite, y) || throw(UnsupportedDataError("MeanRegressor target must contain only finite values."))
    if weights === nothing
        target_mean = mean(y)
    else
        length(weights) == n || throw(SchemaMismatchError(
            "MeanRegressor weights have length $(length(weights)); expected $n observations."))
        all(w -> isfinite(w) && w >= 0, weights) || throw(UnsupportedDataError(
            "MeanRegressor weights must be finite and nonnegative."))
        total = sum(weights)
        total > 0 || throw(UnsupportedDataError("MeanRegressor weights must have a positive sum."))
        target_mean = sum(y .* weights) / total
    end
    schema = infer_schema(X)
    details = (weighted=weights !== nothing, target_mean=target_mean)
    FittedMeanRegressor(model, target_mean,
        FitReport(observations=n, features=size(X, 2), backend=:cpu, details=details), schema)
end

function predict(fitted::FittedMeanRegressor, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "MeanRegressor")
    fill(fitted.mean, size(X, 1))
end
report(fitted::FittedMeanRegressor) = fitted.report
