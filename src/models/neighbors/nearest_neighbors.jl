abstract type AbstractNeighborsEstimator <: AbstractEstimator end

struct NearestNeighbors <: AbstractNeighborsEstimator
    n_neighbors::Int
    metric::Symbol
    function NearestNeighbors(; n_neighbors::Integer=5, metric::Symbol=:euclidean)
        n_neighbors > 0 || throw(InvalidHyperparameterError("n_neighbors must be positive."))
        metric in (:euclidean, :squared_euclidean, :manhattan, :cosine) ||
            throw(InvalidHyperparameterError("Unsupported neighbor metric $metric."))
        new(Int(n_neighbors), metric)
    end
end

"""Classification by uniform or distance-weighted votes of exact nearest neighbors."""
struct KNeighborsClassifier <: AbstractPredictor
    n_neighbors::Int
    metric::Symbol
    weights::Symbol
    function KNeighborsClassifier(; n_neighbors::Integer=5, metric::Symbol=:euclidean,
                                  weights::Symbol=:uniform)
        base = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        weights in (:uniform, :distance) || throw(InvalidHyperparameterError(
            "KNeighborsClassifier weights must be :uniform or :distance."))
        new(base.n_neighbors, base.metric, weights)
    end
end


"""Regression by uniform or distance-weighted averages of exact nearest neighbors."""
struct KNeighborsRegressor <: AbstractPredictor
    n_neighbors::Int
    metric::Symbol
    weights::Symbol
    function KNeighborsRegressor(; n_neighbors::Integer=5, metric::Symbol=:euclidean,
                                 weights::Symbol=:uniform)
        base = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        weights in (:uniform, :distance) || throw(InvalidHyperparameterError(
            "KNeighborsRegressor weights must be :uniform or :distance."))
        new(base.n_neighbors, base.metric, weights)
    end
end

struct FittedNearestNeighbors{M,T,Y,L,R,S} <: AbstractFittedEstimator
    model::M
    training_data::Matrix{T}
    target::Y
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:NearestNeighbors}) = (task=:neighbors, sparse=false, missing=false,
    weights=false, partial_fit=false, probabilistic=false)
capabilities(::Type{<:KNeighborsClassifier}) = (task=:classification, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=true)
capabilities(::Type{<:KNeighborsRegressor}) = (task=:regression, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _fit_neighbors(model, X, target, classes, context)
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_numeric_matrix(X, name)
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError(
        "$name requires at least one observation and feature."))
    model.n_neighbors <= n || throw(InvalidHyperparameterError(
        "$name requested $(model.n_neighbors) neighbors from only $n training observations."))
    details = (n_neighbors=model.n_neighbors, metric=model.metric,
               weights=hasproperty(model, :weights) ? model.weights : :none)
    FittedNearestNeighbors(model, Matrix{float(eltype(X))}(X), target, classes,
        FitReport(observations=n, features=p, backend=:cpu, details=details), infer_schema(X))
end

fit(model::NearestNeighbors, X::AbstractMatrix; context=default_context()) =
    _fit_neighbors(model, X, nothing, nothing, context)

function fit(model::KNeighborsClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    weights === nothing || throw(UnsupportedDataError(
        "KNeighborsClassifier does not accept training weights; choose distance weighting in the model."))
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "KNeighborsClassifier target has length $(length(y)); expected $(size(X, 1))."))
    classes = _classification_classes(y)
    _fit_neighbors(model, X, copy(y), classes, context)
end

function fit(model::KNeighborsRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    weights === nothing || throw(UnsupportedDataError(
        "KNeighborsRegressor does not accept training weights; choose distance weighting in the model."))
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "KNeighborsRegressor target has length $(length(y)); expected $(size(X, 1))."))
    eltype(y) <: Number && all(isfinite, y) || throw(UnsupportedDataError(
        "KNeighborsRegressor requires a finite numeric target."))
    _fit_neighbors(model, X, copy(y), nothing, context)
end

"""Return `(distances, indices)` for the nearest training observations."""
function kneighbors(fitted::FittedNearestNeighbors, X::AbstractMatrix;
                    n_neighbors::Integer=fitted.model.n_neighbors)
    name = string(nameof(typeof(fitted.model)))
    _validate_numeric_matrix(X, name)
    _validate_feature_count(fitted.schema, X, name)
    0 < n_neighbors <= size(fitted.training_data, 1) || throw(InvalidHyperparameterError(
        "n_neighbors must lie between 1 and $(size(fitted.training_data, 1))."))
    all_distances = Kernels.pairwise_distances(X, fitted.training_data; metric=fitted.model.metric)
    indices = Matrix{Int}(undef, size(X, 1), n_neighbors)
    distances = Matrix{eltype(all_distances)}(undef, size(X, 1), n_neighbors)
    for row in axes(X, 1)
        ordering = partialsortperm(view(all_distances, row, :), 1:n_neighbors)
        indices[row, :] .= ordering
        distances[row, :] .= view(all_distances, row, ordering)
    end
    distances, indices
end

transform(fitted::FittedNearestNeighbors, X::AbstractMatrix) = first(kneighbors(fitted, X))

function _neighbor_weights(distances, weighting)
    weighting === :uniform && return ones(eltype(distances), length(distances))
    zero_indices = findall(iszero, distances)
    isempty(zero_indices) || return [index in zero_indices ? one(eltype(distances)) :
                                     zero(eltype(distances)) for index in eachindex(distances)]
    inv.(distances)
end

function predict_proba(fitted::FittedNearestNeighbors{<:KNeighborsClassifier}, X::AbstractMatrix)
    distances, indices = kneighbors(fitted, X)
    probabilities = zeros(eltype(distances), size(X, 1), length(fitted.classes))
    for row in axes(X, 1)
        local_weights = _neighbor_weights(view(distances, row, :), fitted.model.weights)
        for neighbor in axes(indices, 2)
            class = searchsortedfirst(fitted.classes, fitted.target[indices[row, neighbor]])
            probabilities[row, class] += local_weights[neighbor]
        end
        probabilities[row, :] ./= sum(view(probabilities, row, :))
    end
    probabilities
end

function predict(fitted::FittedNearestNeighbors{<:KNeighborsClassifier}, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

function predict(fitted::FittedNearestNeighbors{<:KNeighborsRegressor}, X::AbstractMatrix)
    distances, indices = kneighbors(fitted, X)
    T = float(promote_type(eltype(distances), eltype(fitted.target)))
    predictions = Vector{T}(undef, size(X, 1))
    for row in axes(X, 1)
        local_weights = _neighbor_weights(view(distances, row, :), fitted.model.weights)
        targets = view(fitted.target, view(indices, row, :))
        predictions[row] = sum(local_weights .* targets) / sum(local_weights)
    end
    predictions
end

report(fitted::FittedNearestNeighbors) = fitted.report
