_inspection_model(fitted::FittedGraph) = last(fitted.fitted_nodes).model
_inspection_model(fitted::AbstractFittedEstimator) = fitted.model

function _inspection_schema(fitted::FittedGraph)
    first(fitted.fitted_nodes).schema
end
_inspection_schema(fitted::AbstractFittedEstimator) = fitted.schema

function _default_inspection_score(fitted, targets, predictions)
    task = capabilities(_inspection_model(fitted)).task
    task === :classification && return accuracy_score(targets, predictions)
    task === :regression && return -root_mean_squared_error(targets, predictions)
    throw(ArgumentError(
        "permutation_importance requires an explicit scoring function for task $task."))
end

"""Estimate feature importance by deterministic repeated column permutation.

The default score is accuracy for classifiers and negative RMSE for regressors,
so a positive importance always means that permutation degraded predictive
quality. Custom scoring functions receive `(targets, predictions)`.
"""
function permutation_importance(fitted::AbstractFittedEstimator,
                                X::AbstractMatrix, targets::AbstractVector;
                                scoring=nothing, greater_is_better::Bool=true,
                                n_repeats::Integer=5,
                                context=default_context())
    n_repeats > 0 || throw(ArgumentError(
        "permutation_importance n_repeats must be positive."))
    size(X, 1) == length(targets) || throw(DimensionMismatch(
        "permutation_importance targets must match observation count."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(ArgumentError(
        "permutation_importance requires observations and features."))
    _validate_numeric_matrix(X, "permutation_importance")
    scorer = scoring === nothing ?
        ((truth, prediction) -> _default_inspection_score(
            fitted, truth, prediction)) : scoring
    baseline = scorer(targets, predict(fitted, X))
    baseline isa Real && isfinite(baseline) || throw(ArgumentError(
        "permutation_importance scoring must return a finite real value."))
    T = float(promote_type(typeof(baseline), eltype(X)))
    importances = Matrix{T}(undef, size(X, 2), Int(n_repeats))
    permuted = Matrix{eltype(X)}(X)
    direction = greater_is_better ? one(T) : -one(T)
    for feature in axes(X, 2), repetition in 1:n_repeats
        feature_context = derive_context(
            context, :permutation_importance, feature, repetition)
        ordering = randperm(feature_context.rng, size(X, 1))
        permuted[:, feature] .= view(X, ordering, feature)
        permuted_score = scorer(targets, predict(fitted, permuted))
        permuted_score isa Real && isfinite(permuted_score) || throw(ArgumentError(
            "permutation_importance scoring must return finite real values."))
        importances[feature, repetition] =
            direction * (T(baseline) - T(permuted_score))
        permuted[:, feature] .= view(X, :, feature)
    end
    means = vec(mean(importances; dims=2))
    deviations = vec(std(importances; dims=2, corrected=false))
    schema = _inspection_schema(fitted)
    names = [column.name for column in schema.columns]
    PermutationImportanceResult(T(baseline), importances, means, deviations, names)
end

function permutation_importance(fitted::AbstractFittedEstimator, source,
                                targets::AbstractVector; kwargs...)
    matrix, schema = _numeric_table_input(source)
    result = permutation_importance(fitted, matrix, targets; kwargs...)
    names = [column.name for column in schema.columns]
    PermutationImportanceResult(result.baseline_score, result.importances,
        result.mean_importance, result.standard_deviation, names)
end

function permutation_importance(fitted::AbstractFittedEstimator, dataset::Dataset;
                                kwargs...)
    dataset.target === nothing && throw(ArgumentError(
        "permutation_importance requires a Dataset target."))
    permutation_importance(fitted, dataset.features, dataset.target; kwargs...)
end
