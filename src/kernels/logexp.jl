"""Compute `log(sum(exp(values)))` using a maximum shift."""
function logsumexp(values::AbstractArray; dims=:)
    if dims === Colon() || dims === (:)
        isempty(values) && return -Inf
        maximum_value = maximum(values)
        maximum_value == Inf && return maximum_value
        maximum_value == -Inf && return maximum_value
        return maximum_value + log(sum(exp.(values .- maximum_value)))
    end
    maximum_values = maximum(values; dims=dims)
    shifted = values .- maximum_values
    result = maximum_values .+ log.(sum(exp.(shifted); dims=dims))
    map!((r, m) -> isinf(m) ? m : r, result, result, maximum_values)
end

"""Clamp scalar or array values to the closed interval `[lower, upper]`."""
function clip_values(values, lower, upper)
    lower <= upper || throw(ArgumentError("clip lower bound must not exceed upper bound."))
    values isa AbstractArray ? clamp.(values, lower, upper) : clamp(values, lower, upper)
end

"""Apply a numerically stable softmax along `dims` (rows by default)."""
function softmax(values::AbstractArray; dims=ndims(values))
    maximum_values = maximum(values; dims=dims)
    exponentials = exp.(values .- maximum_values)
    exponentials ./ sum(exponentials; dims=dims)
end

"""Apply a numerically stable log-softmax along `dims`."""
logsoftmax(values::AbstractArray; dims=ndims(values)) = values .- logsumexp(values; dims=dims)

"""Numerically stable logistic sigmoid."""
function sigmoid(value::Real)
    value >= zero(value) ? inv(one(value) + exp(-value)) : begin
        exponential = exp(value)
        exponential / (one(value) + exponential)
    end
end
sigmoid(values::AbstractArray) = sigmoid.(values)

"""Stable binary cross-entropy from logits, optionally reduced."""
function binary_cross_entropy(logits, targets; reduction::Symbol=:mean)
    size(logits) == size(targets) || throw(DimensionMismatch("logits and targets must have equal shapes."))
    all(target -> zero(target) <= target <= one(target), targets) ||
        throw(ArgumentError("binary targets must lie in [0, 1]."))
    losses = max.(logits, zero(eltype(logits))) .- logits .* targets .+ log1p.(exp.(-abs.(logits)))
    reduction === :none && return losses
    reduction === :sum && return sum(losses)
    reduction === :mean && return mean(losses)
    throw(ArgumentError("reduction must be :none, :sum, or :mean."))
end
