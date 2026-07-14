function _validate_classification_vectors(targets, predictions)
    length(targets) == length(predictions) || throw(DimensionMismatch(
        "targets have length $(length(targets)); predictions have length $(length(predictions))."))
    isempty(targets) && throw(ArgumentError("classification metrics require at least one observation."))
end

function _metric_classes(targets, predictions, labels)
    labels === nothing ? sort!(unique(vcat(collect(targets), collect(predictions)))) : collect(labels)
end

"""Fraction (or weighted fraction) of exactly correct predictions."""
function accuracy_score(targets, predictions; weights=nothing)
    _validate_classification_vectors(targets, predictions)
    correct = targets .== predictions
    weights === nothing ? mean(correct) : Kernels.weighted_mean(correct, weights)
end

"""Accumulate a confusion matrix with actual classes in rows and predictions in columns."""
function confusion_matrix(targets, predictions; labels=nothing, weights=nothing)
    _validate_classification_vectors(targets, predictions)
    classes = _metric_classes(targets, predictions, labels)
    length(unique(classes)) == length(classes) || throw(ArgumentError("confusion-matrix labels must be unique."))
    lookup = Dict(class => index for (index, class) in enumerate(classes))
    all(label -> haskey(lookup, label), targets) && all(label -> haskey(lookup, label), predictions) ||
        throw(ArgumentError("targets and predictions must occur in the requested labels."))
    T = weights === nothing ? Int : float(eltype(weights))
    matrix = zeros(T, length(classes), length(classes))
    if weights === nothing
        for (target, prediction) in zip(targets, predictions)
            matrix[lookup[target], lookup[prediction]] += 1
        end
    else
        length(weights) == length(targets) || throw(DimensionMismatch("weights and targets must have equal lengths."))
        all(weight -> isfinite(weight) && weight >= 0, weights) || throw(ArgumentError(
            "classification weights must be finite and nonnegative."))
        for (target, prediction, weight) in zip(targets, predictions, weights)
            matrix[lookup[target], lookup[prediction]] += weight
        end
    end
    ConfusionMatrix(matrix, classes)
end

function _class_statistics(targets, predictions; labels=nothing, weights=nothing)
    result = confusion_matrix(targets, predictions; labels=labels, weights=weights)
    true_positive = diag(result.matrix)
    predicted_positive = vec(sum(result.matrix; dims=1))
    actual_positive = vec(sum(result.matrix; dims=2))
    result.labels, true_positive, predicted_positive, actual_positive
end

_safe_ratio(numerator, denominator, zero_division) =
    iszero(denominator) ? zero_division : numerator / denominator

function _average_scores(scores, support, average)
    average === :none && return scores
    average === :macro && return mean(scores)
    average === :weighted && return iszero(sum(support)) ? zero(eltype(scores)) :
        sum(scores .* support) / sum(support)
    throw(ArgumentError("average must be :none, :macro, :micro, or :weighted."))
end

"""Per-class or averaged precision for binary and multiclass targets."""
function precision_score(targets, predictions; labels=nothing, average::Symbol=:macro, zero_division=0.0, weights=nothing)
    _, true_positive, predicted_positive, actual_positive =
        _class_statistics(targets, predictions; labels=labels, weights=weights)
    average === :micro && return _safe_ratio(sum(true_positive), sum(predicted_positive), zero_division)
    scores = [_safe_ratio(tp, pp, zero_division) for (tp, pp) in zip(true_positive, predicted_positive)]
    _average_scores(scores, actual_positive, average)
end

"""Per-class or averaged recall for binary and multiclass targets."""
function recall_score(targets, predictions; labels=nothing, average::Symbol=:macro, zero_division=0.0, weights=nothing)
    _, true_positive, predicted_positive, actual_positive =
        _class_statistics(targets, predictions; labels=labels, weights=weights)
    average === :micro && return _safe_ratio(sum(true_positive), sum(actual_positive), zero_division)
    scores = [_safe_ratio(tp, ap, zero_division) for (tp, ap) in zip(true_positive, actual_positive)]
    _average_scores(scores, actual_positive, average)
end

"""Per-class or averaged harmonic mean of precision and recall."""
function f1_score(targets, predictions; labels=nothing, average::Symbol=:macro, zero_division=0.0, weights=nothing)
    _, true_positive, predicted_positive, actual_positive =
        _class_statistics(targets, predictions; labels=labels, weights=weights)
    if average === :micro
        precision = _safe_ratio(sum(true_positive), sum(predicted_positive), zero_division)
        recall = _safe_ratio(sum(true_positive), sum(actual_positive), zero_division)
        return _safe_ratio(2precision * recall, precision + recall, zero_division)
    end
    scores = [_safe_ratio(2tp, pp + ap, zero_division)
              for (tp, pp, ap) in zip(true_positive, predicted_positive, actual_positive)]
    _average_scores(scores, actual_positive, average)
end

"""Multiclass log loss using the fitted model's explicit class-column order."""
function log_loss(targets, probabilities::AbstractMatrix; labels=nothing)
    length(targets) == size(probabilities, 1) || throw(DimensionMismatch(
        "targets and probability rows must agree."))
    classes = labels === nothing ? sort!(unique(targets)) : collect(labels)
    length(classes) == size(probabilities, 2) || throw(DimensionMismatch(
        "probability columns must equal the number of class labels."))
    lookup = Dict(class => index for (index, class) in enumerate(classes))
    all(target -> haskey(lookup, target), targets) || throw(ArgumentError(
        "every target must occur in labels."))
    Kernels.log_loss([lookup[target] for target in targets], probabilities)
end
