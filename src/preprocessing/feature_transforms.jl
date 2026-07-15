"""Scale each feature into a fixed interval using training-set extrema."""
struct MinMaxScale{T} <: AbstractTransformer
    feature_range::Tuple{T,T}
    clip::Bool
    function MinMaxScale(; feature_range=(0.0, 1.0), clip::Bool=false)
        length(feature_range) == 2 || throw(InvalidHyperparameterError(
            "MinMaxScale feature_range must contain exactly two values."))
        lower, upper = promote(feature_range...)
        lower isa Real && upper isa Real && isfinite(lower) && isfinite(upper) && lower < upper ||
            throw(InvalidHyperparameterError(
                "MinMaxScale feature_range must be finite and strictly increasing."))
        new{typeof(lower)}((lower, upper), clip)
    end
end

struct FittedMinMaxScale{M,V,R,S} <: AbstractFittedTransformer
    model::M
    minima::V
    ranges::V
    report::R
    schema::S
end

capabilities(::Type{<:MinMaxScale}) = (
    task=:transformation, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function fit(model::MinMaxScale, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "MinMaxScale fitting")
    _validate_numeric_matrix(X, "MinMaxScale")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "MinMaxScale requires at least one observation and feature."))
    T = eltype(X) <: AbstractFloat ? eltype(X) : context.numerics.float_type
    minima = T.(vec(minimum(X; dims=1)))
    maxima = T.(vec(maximum(X; dims=1)))
    ranges = maxima .- minima
    details = (feature_range=model.feature_range, clip=model.clip,
               constant_features=count(iszero, ranges))
    FittedMinMaxScale(model, minima, ranges,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedMinMaxScale, X::AbstractMatrix)
    _validate_numeric_matrix(X, "MinMaxScale")
    _validate_feature_count(fitted.schema, X, "MinMaxScale")
    T = promote_type(float(eltype(X)), eltype(fitted.minima))
    lower, upper = T.(fitted.model.feature_range)
    safe_ranges = map(value -> iszero(value) ? one(T) : T(value), fitted.ranges)
    result = (T.(X) .- transpose(T.(fitted.minima))) ./ transpose(safe_ranges)
    result .= lower .+ (upper - lower) .* result
    fitted.model.clip && clamp!(result, lower, upper)
    result
end

function inverse_transform(fitted::FittedMinMaxScale, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "MinMaxScale")
    T = promote_type(float(eltype(X)), eltype(fitted.minima))
    lower, upper = T.(fitted.model.feature_range)
    safe_ranges = map(value -> iszero(value) ? one(T) : T(value), fitted.ranges)
    ((T.(X) .- lower) ./ (upper - lower)) .* transpose(safe_ranges) .+
        transpose(T.(fitted.minima))
end

report(fitted::FittedMinMaxScale) = fitted.report

"""Center features by their median and scale them by an interquantile range."""
struct RobustScale{T} <: AbstractTransformer
    center::Bool
    scale::Bool
    quantile_range::Tuple{T,T}
    function RobustScale(; center::Bool=true, scale::Bool=true,
                         quantile_range=(0.25, 0.75))
        length(quantile_range) == 2 || throw(InvalidHyperparameterError(
            "RobustScale quantile_range must contain exactly two probabilities."))
        lower, upper = promote(quantile_range...)
        lower isa Real && upper isa Real && isfinite(lower) && isfinite(upper) &&
            0 <= lower < upper <= 1 || throw(InvalidHyperparameterError(
                "RobustScale quantile_range must be strictly increasing within [0, 1]."))
        new{typeof(lower)}(center, scale, (lower, upper))
    end
end

struct FittedRobustScale{M,V,R,S} <: AbstractFittedTransformer
    model::M
    medians::V
    scales::V
    report::R
    schema::S
end

capabilities(::Type{<:RobustScale}) = (
    task=:transformation, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function fit(model::RobustScale, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "RobustScale fitting")
    _validate_numeric_matrix(X, "RobustScale")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "RobustScale requires at least one observation and feature."))
    T = eltype(X) <: AbstractFloat ? eltype(X) : context.numerics.float_type
    lower, upper = model.quantile_range
    medians = model.center ? T[quantile(view(X, :, feature), 0.5)
                               for feature in axes(X, 2)] : zeros(T, size(X, 2))
    raw_scales = model.scale ?
        T[quantile(view(X, :, feature), upper) -
          quantile(view(X, :, feature), lower) for feature in axes(X, 2)] :
        ones(T, size(X, 2))
    scales = map(value -> iszero(value) ? one(T) : value, raw_scales)
    details = (center=model.center, scale=model.scale,
               quantile_range=model.quantile_range,
               zero_ranges=count(iszero, raw_scales))
    FittedRobustScale(model, medians, scales,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedRobustScale, X::AbstractMatrix)
    _validate_numeric_matrix(X, "RobustScale")
    _validate_feature_count(fitted.schema, X, "RobustScale")
    T = promote_type(float(eltype(X)), eltype(fitted.medians))
    (T.(X) .- transpose(T.(fitted.medians))) ./ transpose(T.(fitted.scales))
end

function inverse_transform(fitted::FittedRobustScale, X::AbstractMatrix)
    _validate_feature_count(fitted.schema, X, "RobustScale")
    T = promote_type(float(eltype(X)), eltype(fitted.medians))
    T.(X) .* transpose(T.(fitted.scales)) .+ transpose(T.(fitted.medians))
end

report(fitted::FittedRobustScale) = fitted.report

"""Normalize each observation to unit L1, L2, or maximum norm."""
struct Normalize <: AbstractTransformer
    norm::Symbol
    function Normalize(; norm::Symbol=:l2)
        norm in (:l1, :l2, :max) || throw(InvalidHyperparameterError(
            "Normalize norm must be :l1, :l2, or :max."))
        new(norm)
    end
end

struct FittedNormalize{M,R,S} <: AbstractFittedTransformer
    model::M
    report::R
    schema::S
end

capabilities(::Type{<:Normalize}) = (
    task=:transformation, sparse=true, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function fit(model::Normalize, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "Normalize fitting")
    _validate_numeric_matrix(X, "Normalize")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "Normalize requires at least one observation and feature."))
    FittedNormalize(model,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=(norm=model.norm,), context=context), infer_schema(X))
end

function _row_norms(X, norm::Symbol)
    norm === :l1 && return vec(sum(abs, X; dims=2))
    norm === :l2 && return sqrt.(vec(sum(abs2, X; dims=2)))
    vec(maximum(abs, X; dims=2))
end

function transform(fitted::FittedNormalize, X::AbstractMatrix)
    _validate_numeric_matrix(X, "Normalize")
    _validate_feature_count(fitted.schema, X, "Normalize")
    T = float(eltype(X))
    data = T.(X)
    norms = _row_norms(data, fitted.model.norm)
    inverse_norms = map(value -> iszero(value) ? one(T) : inv(T(value)), norms)
    Diagonal(inverse_norms) * data
end

report(fitted::FittedNormalize) = fitted.report

"""Expand features into monomials up to a requested total degree."""
struct PolynomialFeatures <: AbstractTransformer
    degree::Int
    include_bias::Bool
    interaction_only::Bool
    function PolynomialFeatures(; degree::Integer=2, include_bias::Bool=true,
                                interaction_only::Bool=false)
        1 <= degree <= 32 || throw(InvalidHyperparameterError(
            "PolynomialFeatures degree must lie between 1 and 32."))
        new(Int(degree), include_bias, interaction_only)
    end
end


struct FittedPolynomialFeatures{M,T,R,S} <: AbstractFittedTransformer
    model::M
    terms::T
    report::R
    schema::S
end

capabilities(::Type{<:PolynomialFeatures}) = (
    task=:transformation, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function _polynomial_terms(features::Int, degree::Int, interaction_only::Bool,
                           include_bias::Bool; limit::Int=100_000)
    terms = Vector{Vector{Int}}()
    function add_term!(term)
        length(terms) < limit || throw(UnsupportedDataError(
            "PolynomialFeatures exceeds the $limit-column safety limit; reduce degree or input width."))
        push!(terms, term)
    end
    include_bias && add_term!(Int[])
    current = Int[]
    function extend!(start::Int, remaining::Int)
        if remaining == 0
            add_term!(copy(current))
            return
        end
        for feature in start:features
            push!(current, feature)
            extend!(interaction_only ? feature + 1 : feature, remaining - 1)
            pop!(current)
        end
    end
    for total_degree in 1:degree
        interaction_only && total_degree > features && break
        extend!(1, total_degree)
    end
    terms
end

function fit(model::PolynomialFeatures, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "PolynomialFeatures fitting")
    _validate_numeric_matrix(X, "PolynomialFeatures")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "PolynomialFeatures requires at least one observation and feature."))
    terms = _polynomial_terms(size(X, 2), model.degree,
                              model.interaction_only, model.include_bias)
    details = (degree=model.degree, include_bias=model.include_bias,
               interaction_only=model.interaction_only, output_features=length(terms))
    FittedPolynomialFeatures(model, terms,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedPolynomialFeatures, X::AbstractMatrix)
    _validate_numeric_matrix(X, "PolynomialFeatures")
    _validate_feature_count(fitted.schema, X, "PolynomialFeatures")
    T = float(eltype(X))
    result = Matrix{T}(undef, size(X, 1), length(fitted.terms))
    @inbounds for (column, term) in enumerate(fitted.terms)
        if isempty(term)
            result[:, column] .= one(T)
        else
            for row in axes(X, 1)
                value = one(T)
                for feature in term
                    value *= T(X[row, feature])
                end
                result[row, column] = value
            end
        end
    end
    result
end

report(fitted::FittedPolynomialFeatures) = fitted.report
