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
    partial_fit=false, probabilistic=false,
)

function fit(model::Standardize, X::AbstractMatrix; context=default_context())
    require_cpu(context, "Standardize fitting")
    X isa SparseMatrixCSC && model.center && throw(UnsupportedDataError(
        "Standardize with center=true would destroy sparse structure; use center=false."))
    _validate_numeric_matrix(X, "Standardize")
    n = size(X, 1)
    n > 0 || throw(UnsupportedDataError("Standardize requires at least one observation."))
    means = model.center ? vec(mean(X; dims=1)) : zeros(eltype(X), size(X, 2))
    raw_scales = model.scale ? vec(std(X; dims=1, corrected=false)) : ones(eltype(X), size(X, 2))
    scales = map(s -> iszero(s) ? one(s) : s, raw_scales)
    schema = infer_schema(X)
    details = (center=model.center, scale=model.scale, zero_variance=count(iszero, raw_scales))
    FittedStandardize(model, means, scales,
        FitReport(observations=n, features=size(X, 2), backend=:cpu, details=details), schema)
end

function transform(fitted::FittedStandardize, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "Standardize")
    (X .- transpose(fitted.means)) ./ transpose(fitted.scales)
end

inverse_transform(fitted::FittedStandardize, X::AbstractMatrix) =
    X .* transpose(fitted.scales) .+ transpose(fitted.means)
report(fitted::FittedStandardize) = fitted.report
