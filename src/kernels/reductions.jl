function _validate_weights(values, weights; accumulation_type=nothing)
    length(values) == length(weights) || throw(DimensionMismatch(
        "weights have length $(length(weights)); expected $(length(values))."))
    all(w -> isfinite(w) && w >= zero(w), weights) || throw(ArgumentError(
        "weights must be finite and nonnegative."))
    A = accumulation_type === nothing ? float(eltype(weights)) : accumulation_type
    total = stable_sum(weights; accumulation_type=A)
    total > zero(total) || throw(ArgumentError("weights must have a positive sum."))
    total
end

"""Sum a collection, optionally using compensated floating-point accumulation."""
function reduction_sum(values; stable::Bool=false,
                       accumulation_type=float(eltype(values)))
    stable && return stable_sum(values; accumulation_type)
    sum(accumulation_type, values)
end

"""Arithmetic mean using an explicit internal accumulation type."""
function reduction_mean(values; stable::Bool=false,
                        accumulation_type=float(eltype(values)))
    isempty(values) && throw(ArgumentError("mean input cannot be empty."))
    reduction_sum(values; stable, accumulation_type) / accumulation_type(length(values))
end

"""Population or corrected sample variance with explicit accumulation policy."""
function reduction_variance(values; corrected::Bool=false, stable::Bool=false,
                            accumulation_type=float(eltype(values)))
    _reduction_variance(values, corrected, stable, accumulation_type)
end

function _reduction_variance(values, corrected::Bool, stable::Bool,
                             ::Type{A}) where {A<:AbstractFloat}
    count = length(values)
    denominator = count - Int(corrected)
    denominator > 0 || throw(ArgumentError(
        corrected ? "corrected variance requires at least two values." :
                    "variance input cannot be empty."))
    center = (stable ? _stable_sum(values, A) : sum(A, values)) / A(count)
    total = zero(A)
    correction = zero(A)
    @inbounds for value in values
        squared = abs2(A(value) - center)
        if stable
            candidate = total + squared
            if abs(total) >= abs(squared)
                correction += (total - candidate) + squared
            else
                correction += (squared - candidate) + total
            end
            total = candidate
        else
            total += squared
        end
    end
    (total + correction) / A(denominator)
end

"""Return `(minimum, maximum)` and reject an empty reduction explicitly."""
function extrema_values(values)
    isempty(values) && throw(ArgumentError("extrema input cannot be empty."))
    extrema(values)
end

"""Index of the first minimum, giving deterministic tie handling."""
function argmin_index(values)
    isempty(values) && throw(ArgumentError("argmin input cannot be empty."))
    argmin(values)
end

"""Index of the first maximum, giving deterministic tie handling."""
function argmax_index(values)
    isempty(values) && throw(ArgumentError("argmax input cannot be empty."))
    argmax(values)
end

"""Sum values with Neumaier compensation in the requested accumulation type."""
stable_sum(values; accumulation_type=float(eltype(values))) =
    _stable_sum(values, accumulation_type)

function _stable_sum(values, ::Type{A}) where {A<:AbstractFloat}
    total = zero(A)
    correction = zero(A)
    @inbounds for value in values
        converted = A(value)
        candidate = total + converted
        if abs(total) >= abs(converted)
            correction += (total - candidate) + converted
        else
            correction += (converted - candidate) + total
        end
        total = candidate
    end
    total + correction
end

"""Return `sum(values .* weights)` after validating frequency weights."""
function weighted_sum(values, weights; stable::Bool=false,
                      accumulation_type=float(promote_type(eltype(values), eltype(weights))))
    _weighted_sum(values, weights, stable, accumulation_type)
end

function _weighted_sum(values, weights, stable::Bool, ::Type{A}) where {A<:AbstractFloat}
    _validate_weights(values, weights; accumulation_type=A)
    if !stable
        return sum(A(values[index]) * A(weights[index]) for index in eachindex(values, weights))
    end
    total = zero(A)
    correction = zero(A)
    @inbounds for index in eachindex(values, weights)
        value = A(values[index]) * A(weights[index])
        candidate = total + value
        if abs(total) >= abs(value)
            correction += (total - candidate) + value
        else
            correction += (value - candidate) + total
        end
        total = candidate
    end
    total + correction
end

"""Return the nonnegative-weighted arithmetic mean."""
function weighted_mean(values, weights; stable::Bool=false,
                       accumulation_type=float(promote_type(eltype(values), eltype(weights))))
    _weighted_mean(values, weights, stable, accumulation_type)
end

function _weighted_mean(values, weights, stable::Bool, ::Type{A}) where {A<:AbstractFloat}
    total = _validate_weights(values, weights; accumulation_type=A)
    _weighted_sum(values, weights, stable, A) / total
end

"""
Return weighted variance. The default is population variance; `corrected=true`
uses the reliability-weight correction `sum(w) - sum(w²)/sum(w)`.
"""
function weighted_variance(values, weights; corrected::Bool=false, mean_value=nothing)
    total = _validate_weights(values, weights)
    center = mean_value === nothing ? sum(values .* weights) / total : mean_value
    numerator = sum(weights .* abs2.(values .- center))
    denominator = corrected ? total - sum(abs2, weights) / total : total
    denominator > zero(denominator) || throw(ArgumentError(
        "corrected weighted variance requires at least two positive effective observations."))
    numerator / denominator
end

"""Compute a scaled Euclidean norm without avoidable overflow or underflow."""
function stable_norm(values)
    scale = maximum(abs, values; init=zero(eltype(values)))
    iszero(scale) && return float(scale)
    scale * sqrt(sum(abs2, values ./ scale))
end
