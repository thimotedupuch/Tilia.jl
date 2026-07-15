struct Standardize <: AbstractTransformer
    center::Bool
    scale::Bool
end
Standardize(; center=true, scale=true) = Standardize(center, scale)

abstract type AbstractFittedTransformer <: AbstractFittedEstimator end

struct FittedStandardize{M,V,S,R,SC} <: AbstractFittedTransformer
    model::M
    means::V
    scales::S
    report::R
    schema::SC
end

capabilities(::Type{<:Standardize}) = (
    task=:transformation, sparse=false, missing=false, weights=false,
    partial_fit=true, probabilistic=false,
)

function fit(model::Standardize, X::AbstractMatrix; context=default_context())
    require_cpu(context, "Standardize fitting")
    X isa SparseMatrixCSC && model.center && throw(UnsupportedDataError(
        "Standardize with center=true would destroy sparse structure; use center=false."))
    _validate_numeric_matrix(X, "Standardize")
    n = size(X, 1)
    n > 0 || throw(UnsupportedDataError("Standardize requires at least one observation."))
    running_mean = vec(mean(X; dims=1))
    m2 = vec(sum(abs2, X .- transpose(running_mean); dims=1))
    means = model.center ? running_mean : zeros(eltype(X), size(X, 2))
    raw_scales = model.scale ? sqrt.(m2 ./ n) : ones(eltype(X), size(X, 2))
    scales = map(s -> iszero(s) ? one(s) : s, raw_scales)
    schema = infer_schema(X)
    details = (center=model.center, scale=model.scale,
               zero_variance=count(iszero, raw_scales), count=n,
               running_mean=running_mean, m2=m2, partial_updates=0)
    FittedStandardize(model, means, scales,
        FitReport(observations=n, features=size(X, 2), backend=:cpu,
                  details=details, context=context), schema)
end

partial_fit(model::Standardize, X::AbstractMatrix; kwargs...) = fit(model, X; kwargs...)

"""Merge another batch into fitted population mean and variance statistics."""
function partial_fit(fitted::FittedStandardize, X::AbstractMatrix;
                     context=default_context())
    require_cpu(context, "Standardize partial fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError(
        "Standardize partial_fit currently requires dense batches."))
    _validate_feature_count(fitted.schema, X, "Standardize")
    _validate_numeric_matrix(X, "Standardize")
    batch_count = size(X, 1)
    batch_count > 0 || throw(UnsupportedDataError(
        "Standardize partial_fit requires at least one observation."))
    previous_count = fitted.report.details.count
    previous_mean = fitted.report.details.running_mean
    previous_m2 = fitted.report.details.m2
    batch_mean = vec(mean(X; dims=1))
    batch_m2 = vec(sum(abs2, X .- transpose(batch_mean); dims=1))
    total_count = previous_count + batch_count
    delta = batch_mean .- previous_mean
    running_mean = previous_mean .+ delta .* (batch_count / total_count)
    m2 = previous_m2 .+ batch_m2 .+
         abs2.(delta) .* (previous_count * batch_count / total_count)
    means = fitted.model.center ? running_mean : zeros(eltype(running_mean), length(running_mean))
    raw_scales = fitted.model.scale ? sqrt.(m2 ./ total_count) :
                 ones(eltype(running_mean), length(running_mean))
    scales = map(value -> iszero(value) ? one(value) : value, raw_scales)
    details = (center=fitted.model.center, scale=fitted.model.scale,
               zero_variance=count(iszero, raw_scales), count=total_count,
               running_mean=running_mean, m2=m2,
               partial_updates=fitted.report.details.partial_updates + 1)
    updated_report = FitReport(observations=total_count,
        features=nfeatures(fitted.schema), backend=:cpu, details=details,
        context=context)
    FittedStandardize(fitted.model, means, scales, updated_report, fitted.schema)
end

function transform(fitted::FittedStandardize, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "Standardize")
    (X .- transpose(fitted.means)) ./ transpose(fitted.scales)
end

inverse_transform(fitted::FittedStandardize, X::AbstractMatrix) =
    X .* transpose(fitted.scales) .+ transpose(fitted.means)
report(fitted::FittedStandardize) = fitted.report
