function _validate_weights(values, weights)
    length(values) == length(weights) || throw(DimensionMismatch(
        "weights have length $(length(weights)); expected $(length(values))."))
    all(w -> isfinite(w) && w >= zero(w), weights) || throw(ArgumentError(
        "weights must be finite and nonnegative."))
    total = sum(weights)
    total > zero(total) || throw(ArgumentError("weights must have a positive sum."))
    total
end

"""Return `sum(values .* weights)` after validating frequency weights."""
function weighted_sum(values, weights)
    _validate_weights(values, weights)
    sum(values .* weights)
end

"""Return the nonnegative-weighted arithmetic mean."""
function weighted_mean(values, weights)
    total = _validate_weights(values, weights)
    sum(values .* weights) / total
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
