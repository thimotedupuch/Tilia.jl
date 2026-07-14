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
