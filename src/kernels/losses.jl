function _check_predictions(targets, predictions)
    size(targets) == size(predictions) || throw(DimensionMismatch(
        "targets and predictions must have equal shapes."))
    isempty(targets) && throw(ArgumentError("losses require at least one observation."))
end

function mean_squared_error(targets, predictions; weights=nothing)
    _check_predictions(targets, predictions)
    errors = abs2.(predictions .- targets)
    weights === nothing ? mean(errors) : weighted_mean(errors, weights)
end

root_mean_squared_error(targets, predictions; weights=nothing) =
    sqrt(mean_squared_error(targets, predictions; weights=weights))

function mean_absolute_error(targets, predictions; weights=nothing)
    _check_predictions(targets, predictions)
    errors = abs.(predictions .- targets)
    weights === nothing ? mean(errors) : weighted_mean(errors, weights)
end

function huber_loss(targets, predictions; delta=1.35, weights=nothing)
    _check_predictions(targets, predictions)
    delta > 0 || throw(ArgumentError("Huber loss delta must be positive."))
    diffs = abs.(predictions .- targets)
    errors = map(d -> d <= delta ? 0.5 * d^2 : delta * (d - 0.5 * delta), diffs)
    weights === nothing ? mean(errors) : weighted_mean(errors, weights)
end

function quantile_loss(targets, predictions; quantile=0.5, weights=nothing)
    _check_predictions(targets, predictions)
    zero(quantile) <= quantile <= one(quantile) || throw(ArgumentError("quantile must be in [0, 1]."))
    diffs = targets .- predictions
    errors = map(d -> d >= zero(d) ? quantile * d : (quantile - one(quantile)) * d, diffs)
    weights === nothing ? mean(errors) : weighted_mean(errors, weights)
end

"""Multiclass logarithmic loss for an observation-by-class probability matrix."""
function log_loss(labels::AbstractVector{<:Integer}, probabilities::AbstractMatrix; epsilon=nothing)
    length(labels) == size(probabilities, 1) || throw(DimensionMismatch(
        "labels have length $(length(labels)); probabilities have $(size(probabilities, 1)) rows."))
    isempty(labels) && throw(ArgumentError("log loss requires at least one observation."))
    classes = size(probabilities, 2)
    all(label -> 1 <= label <= classes, labels) || throw(ArgumentError(
        "labels must be integer class indices in 1:$classes."))
    all(isfinite, probabilities) || throw(ArgumentError("probabilities must be finite."))
    all(probability -> zero(probability) <= probability <= one(probability), probabilities) ||
        throw(ArgumentError("probabilities must lie in [0, 1]."))
    tolerance = sqrt(eps(float(eltype(probabilities))))
    all(abs.(vec(sum(probabilities; dims=2)) .- one(eltype(probabilities))) .<= tolerance) ||
        throw(ArgumentError("each probability row must sum to one."))
    clipping = epsilon === nothing ? eps(float(eltype(probabilities))) : epsilon
    -mean(log(max(probabilities[index, label], clipping)) for (index, label) in enumerate(labels))
end
