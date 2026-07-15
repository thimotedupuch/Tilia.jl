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

function _binary_curve_inputs(targets, scores, positive_label, weights)
    length(targets) == length(scores) || throw(DimensionMismatch(
        "targets and scores must have equal lengths."))
    isempty(targets) && throw(ArgumentError(
        "binary curves require at least one observation."))
    all(isfinite, scores) || throw(ArgumentError("curve scores must be finite."))
    classes = sort!(unique(targets))
    length(classes) == 2 || throw(ArgumentError(
        "binary curves require exactly two observed classes."))
    positive = positive_label === nothing ? classes[end] : positive_label
    positive in classes || throw(ArgumentError(
        "positive_label must occur in targets."))
    T = float(promote_type(eltype(scores),
        weights === nothing ? eltype(scores) : eltype(weights)))
    observation_weights = weights === nothing ? ones(T, length(scores)) : T.(weights)
    length(observation_weights) == length(scores) || throw(DimensionMismatch(
        "curve weights and scores must have equal lengths."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) ||
        throw(ArgumentError("curve weights must be finite and nonnegative."))
    positives = targets .== positive
    positive_weight = sum(observation_weights[positives])
    negative_weight = sum(observation_weights[.!positives])
    positive_weight > 0 && negative_weight > 0 || throw(ArgumentError(
        "both binary classes must have positive total weight."))
    T.(scores), positives, observation_weights, T(positive_weight), T(negative_weight)
end

function _binary_score_groups(scores)
    ordering = sortperm(eachindex(scores); by=index -> (-scores[index], index))
    groups = UnitRange{Int}[]
    start = 1
    while start <= length(ordering)
        stop = start
        while stop < length(ordering) && scores[ordering[stop + 1]] == scores[ordering[start]]
            stop += 1
        end
        push!(groups, start:stop)
        start = stop + 1
    end
    ordering, groups
end

"""Binary receiver-operating-characteristic curve with descending thresholds."""
function roc_curve(targets, scores::AbstractVector; positive_label=nothing,
                   weights=nothing)
    values, positives, observation_weights, total_positive, total_negative =
        _binary_curve_inputs(targets, scores, positive_label, weights)
    ordering, groups = _binary_score_groups(values)
    T = eltype(values)
    false_positive_rate = T[zero(T)]
    true_positive_rate = T[zero(T)]
    thresholds = T[T(Inf)]
    true_positive = zero(T)
    false_positive = zero(T)
    for group in groups
        for position in group
            index = ordering[position]
            if positives[index]
                true_positive += observation_weights[index]
            else
                false_positive += observation_weights[index]
            end
        end
        push!(true_positive_rate, true_positive / total_positive)
        push!(false_positive_rate, false_positive / total_negative)
        push!(thresholds, values[ordering[first(group)]])
    end
    ROCResult(false_positive_rate, true_positive_rate, thresholds)
end

"""Binary precision–recall curve, starting at recall zero and threshold `Inf`."""
function precision_recall_curve(targets, scores::AbstractVector;
                                positive_label=nothing, weights=nothing)
    values, positives, observation_weights, total_positive, _ =
        _binary_curve_inputs(targets, scores, positive_label, weights)
    ordering, groups = _binary_score_groups(values)
    T = eltype(values)
    precision = T[one(T)]
    recall = T[zero(T)]
    thresholds = T[T(Inf)]
    true_positive = zero(T)
    predicted_positive = zero(T)
    for group in groups
        for position in group
            index = ordering[position]
            weight = observation_weights[index]
            predicted_positive += weight
            positives[index] && (true_positive += weight)
        end
        push!(precision, true_positive / predicted_positive)
        push!(recall, true_positive / total_positive)
        push!(thresholds, values[ordering[first(group)]])
    end
    PrecisionRecallResult(precision, recall, thresholds)
end

"""Trapezoidal area for a curve whose x coordinates are monotone increasing."""
function area_under_curve(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch(
        "curve coordinates must have equal lengths."))
    length(x) >= 2 || throw(ArgumentError("area calculation requires at least two points."))
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError(
        "area calculation requires finite coordinates."))
    all(diff(x) .>= 0) || throw(ArgumentError(
        "curve x coordinates must be monotone increasing."))
    sum((x[index + 1] - x[index]) * (y[index + 1] + y[index]) / 2
        for index in 1:length(x)-1)
end

area_under_curve(result::ROCResult) =
    area_under_curve(result.false_positive_rate, result.true_positive_rate)
area_under_curve(result::PrecisionRecallResult) =
    area_under_curve(result.recall, result.precision)

"""Reliability diagram data for binary probabilistic predictions."""
function calibration_curve(targets, probabilities::AbstractVector;
                           positive_label=nothing, n_bins::Integer=10,
                           strategy::Symbol=:uniform, weights=nothing)
    n_bins > 0 || throw(ArgumentError("calibration n_bins must be positive."))
    strategy in (:uniform, :quantile) || throw(ArgumentError(
        "calibration strategy must be :uniform or :quantile."))
    values, positives, observation_weights, _, _ =
        _binary_curve_inputs(targets, probabilities, positive_label, weights)
    all(value -> 0 <= value <= 1, values) || throw(ArgumentError(
        "calibration probabilities must lie in [0, 1]."))
    T = eltype(values)
    edges = if strategy === :uniform
        collect(range(zero(T), one(T); length=Int(n_bins) + 1))
    else
        requested = [quantile(values, probability)
                     for probability in range(0, 1; length=Int(n_bins) + 1)]
        sort!(unique(T[zero(T); requested; one(T)]))
    end
    bin_count = length(edges) - 1
    counts = zeros(T, bin_count)
    probability_sums = zeros(T, bin_count)
    positive_sums = zeros(T, bin_count)
    for index in eachindex(values)
        bin = clamp(searchsortedlast(edges, values[index]), 1, bin_count)
        weight = observation_weights[index]
        counts[bin] += weight
        probability_sums[bin] += weight * values[index]
        positive_sums[bin] += weight * positives[index]
    end
    occupied = findall(>(zero(T)), counts)
    CalibrationResult(probability_sums[occupied] ./ counts[occupied],
                      positive_sums[occupied] ./ counts[occupied],
                      counts[occupied], edges)
end
