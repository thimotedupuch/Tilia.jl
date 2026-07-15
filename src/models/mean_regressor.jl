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
    partial_fit=true, probabilistic=false,
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
    total_weight = weights === nothing ? float(eltype(y))(n) : sum(weights)
    details = (weighted=weights !== nothing, target_mean=target_mean,
               total_weight=total_weight)
    FittedMeanRegressor(model, target_mean,
        FitReport(observations=n, features=size(X, 2), backend=:cpu,
                  details=details, context=context), schema)
end

partial_fit(model::MeanRegressor, X::AbstractMatrix, y::AbstractVector; kwargs...) =
    fit(model, X, y; kwargs...)

"""Update a fitted mean regressor from another observation batch."""
function partial_fit(fitted::FittedMeanRegressor, X::AbstractMatrix, y::AbstractVector;
                     weights=nothing, context=default_context())
    _validate_feature_count(fitted.schema, X, "MeanRegressor")
    batch_context = derive_context(context, :partial_fit, :batch,
                                   fitted.report.observations + 1)
    batch = fit(fitted.model, X, y; weights=weights, context=batch_context)
    previous_weight = hasproperty(fitted.report.details, :total_weight) ?
        fitted.report.details.total_weight : fitted.report.observations
    batch_weight = batch.report.details.total_weight
    total_weight = previous_weight + batch_weight
    updated_mean = (fitted.mean * previous_weight + batch.mean * batch_weight) / total_weight
    details = (weighted=fitted.report.details.weighted || weights !== nothing,
               target_mean=updated_mean, total_weight=total_weight,
               partial_updates=get(fitted.report.details, :partial_updates, 0) + 1)
    updated_report = FitReport(observations=fitted.report.observations + size(X, 1),
        features=nfeatures(fitted.schema), backend=:cpu, details=details,
        context=context)
    FittedMeanRegressor(fitted.model, updated_mean, updated_report, fitted.schema)
end

function predict(fitted::FittedMeanRegressor, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "MeanRegressor")
    fill(fitted.mean, size(X, 1))
end
report(fitted::FittedMeanRegressor) = fitted.report
