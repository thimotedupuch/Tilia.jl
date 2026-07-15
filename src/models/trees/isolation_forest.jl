"""Isolation forest for unsupervised anomaly detection on dense numeric data."""
struct IsolationForest <: AbstractEstimator
    n_estimators::Int
    max_samples::Union{Symbol,Int,Float64}
    contamination::Union{Symbol,Float64}
    max_features::Float64
    function IsolationForest(; n_estimators::Integer=100, max_samples=:auto,
            contamination=:auto, max_features::Real=1.0)
        n_estimators > 0 || throw(InvalidHyperparameterError(
            "IsolationForest n_estimators must be positive."))
        valid_samples = max_samples === :auto ||
            (max_samples isa Integer && max_samples > 1) ||
            (max_samples isa AbstractFloat && 0 < max_samples <= 1)
        valid_samples || throw(InvalidHyperparameterError(
            "max_samples must be :auto, an integer greater than one, or a fraction in (0,1]."))
        valid_contamination = contamination === :auto ||
            (contamination isa AbstractFloat && 0 < contamination <= 0.5)
        valid_contamination || throw(InvalidHyperparameterError(
            "contamination must be :auto or a fraction in (0,0.5]."))
        isfinite(max_features) && 0 < max_features <= 1 || throw(InvalidHyperparameterError(
            "max_features must lie in (0,1]."))
        new(Int(n_estimators), max_samples, contamination, Float64(max_features))
    end
end

struct IsolationNode{T}
    feature::Int
    threshold::T
    left::Int
    right::Int
    samples::Int
    is_leaf::Bool
end

struct IsolationTree{T}
    nodes::Vector{IsolationNode{T}}
    features::Vector{Int}
end

struct FittedIsolationForest{M,T,R,S} <: AbstractFittedEstimator
    model::M
    trees::Vector{IsolationTree{T}}
    sample_size::Int
    threshold::T
    report::R
    schema::S
end

capabilities(::Type{<:IsolationForest}) = (task=:anomaly_detection, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _isolation_sample_size(max_samples, n)
    max_samples === :auto && return min(256, n)
    max_samples isa Integer && return min(Int(max_samples), n)
    max(2, min(n, floor(Int, max_samples * n)))
end

function _build_isolation_tree!(nodes, X, indices, depth, max_depth, rng, T)
    node_index = length(nodes) + 1
    push!(nodes, IsolationNode(0, zero(T), 0, 0, length(indices), true))
    (depth >= max_depth || length(indices) <= 1) && return node_index
    candidates = Int[]
    for feature in axes(X, 2)
        minimum(view(X, indices, feature)) < maximum(view(X, indices, feature)) &&
            push!(candidates, feature)
    end
    isempty(candidates) && return node_index
    feature = rand(rng, candidates)
    minimum_value = minimum(view(X, indices, feature))
    maximum_value = maximum(view(X, indices, feature))
    threshold = minimum_value + rand(rng, T) * (maximum_value - minimum_value)
    left_indices = [index for index in indices if X[index, feature] <= threshold]
    right_indices = [index for index in indices if X[index, feature] > threshold]
    (isempty(left_indices) || isempty(right_indices)) && return node_index
    left = _build_isolation_tree!(nodes, X, left_indices, depth + 1, max_depth, rng, T)
    right = _build_isolation_tree!(nodes, X, right_indices, depth + 1, max_depth, rng, T)
    nodes[node_index] = IsolationNode(feature, threshold, left, right, length(indices), false)
    node_index
end

function _average_unsuccessful_path_length(n)
    n <= 1 && return 0.0
    n == 2 && return 1.0
    2 * (log(n - 1) + Base.MathConstants.eulergamma) - 2 * (n - 1) / n
end

function _isolation_path_length(tree, row)
    index = 1
    depth = 0
    while !tree.nodes[index].is_leaf
        node = tree.nodes[index]
        index = row[node.feature] <= node.threshold ? node.left : node.right
        depth += 1
    end
    depth + _average_unsuccessful_path_length(tree.nodes[index].samples)
end

function _anomaly_scores(trees, X, sample_size, T)
    normalization = T(_average_unsuccessful_path_length(sample_size))
    scores = Vector{T}(undef, size(X, 1))
    for row in axes(X, 1)
        average_path = mean(_isolation_path_length(tree, view(X, row, tree.features))
                            for tree in trees)
        scores[row] = T(2)^(-T(average_path) / normalization)
    end
    scores
end

function fit(model::IsolationForest, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "IsolationForest fitting")
    _validate_numeric_matrix(X, "IsolationForest")
    n, p = size(X)
    n >= 2 && p > 0 || throw(UnsupportedDataError(
        "IsolationForest requires at least two observations and one feature."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    sample_size = _isolation_sample_size(model.max_samples, n)
    feature_count = max(1, floor(Int, model.max_features * p))
    maximum_depth = ceil(Int, log2(sample_size))
    trees = IsolationTree{T}[]
    for tree_index in 1:model.n_estimators
        tree_context = derive_context(context, :isolation_forest, :tree, tree_index)
        sample_indices = randperm(tree_context.rng, n)[1:sample_size]
        features = sort!(randperm(tree_context.rng, p)[1:feature_count])
        nodes = IsolationNode{T}[]
        _build_isolation_tree!(nodes, view(data, :, features), sample_indices,
                               0, maximum_depth, tree_context.rng, T)
        push!(trees, IsolationTree(nodes, features))
    end
    scores = _anomaly_scores(trees, data, sample_size, T)
    threshold = if model.contamination === :auto
        T(0.5)
    else
        ordered = sort(scores)
        ordered[clamp(ceil(Int, (1 - model.contamination) * n), 1, n)]
    end
    details = (n_estimators=model.n_estimators, sample_size=sample_size,
               maximum_depth=maximum_depth, features_per_tree=feature_count,
               contamination=model.contamination, threshold=threshold)
    FittedIsolationForest(model, trees, sample_size, threshold,
        FitReport(observations=n, features=p, backend=:cpu, details=details,
                  context=context), infer_schema(X))
end

"""Return isolation anomaly scores, where larger values indicate stronger anomalies."""
function anomaly_score(fitted::FittedIsolationForest, X::AbstractMatrix)
    _validate_numeric_matrix(X, "IsolationForest")
    _validate_feature_count(fitted.schema, X, "IsolationForest")
    _anomaly_scores(fitted.trees, X, fitted.sample_size, eltype(fitted.threshold))
end

"""Return `-1` for anomalies and `1` for inliers."""
predict(fitted::FittedIsolationForest, X::AbstractMatrix) =
    ifelse.(anomaly_score(fitted, X) .>= fitted.threshold, -1, 1)

report(fitted::FittedIsolationForest) = fitted.report
