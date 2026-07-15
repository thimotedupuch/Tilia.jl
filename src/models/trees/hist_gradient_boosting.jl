abstract type AbstractHistGradientBoosting <: AbstractPredictor end

struct HistGradientBoostingRegressor <: AbstractHistGradientBoosting
    n_estimators::Int
    learning_rate::Float64
    max_depth::Int
    max_bins::Int
    min_samples_leaf::Int
    tolerance::Float64
    l1_regularization::Float64
    l2_regularization::Float64
    row_subsample::Float64
    feature_subsample::Float64
    function HistGradientBoostingRegressor(; n_estimators::Integer=100,
            learning_rate::Real=0.1, max_depth::Integer=3, max_bins::Integer=255,
            min_samples_leaf::Integer=5, tolerance::Real=1e-7,
            l1_regularization::Real=0.0, l2_regularization::Real=0.0,
            row_subsample::Real=1.0, feature_subsample::Real=1.0)
        _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
            max_bins, min_samples_leaf, tolerance, l1_regularization,
            l2_regularization, row_subsample, feature_subsample)
        new(Int(n_estimators), Float64(learning_rate), Int(max_depth),
            Int(max_bins), Int(min_samples_leaf), Float64(tolerance),
            Float64(l1_regularization), Float64(l2_regularization),
            Float64(row_subsample), Float64(feature_subsample))
    end
end

struct HistGradientBoostingClassifier <: AbstractHistGradientBoosting
    n_estimators::Int
    learning_rate::Float64
    max_depth::Int
    max_bins::Int
    min_samples_leaf::Int
    tolerance::Float64
    l1_regularization::Float64
    l2_regularization::Float64
    row_subsample::Float64
    feature_subsample::Float64
    function HistGradientBoostingClassifier(; n_estimators::Integer=100,
            learning_rate::Real=0.1, max_depth::Integer=3, max_bins::Integer=255,
            min_samples_leaf::Integer=5, tolerance::Real=1e-7,
            l1_regularization::Real=0.0, l2_regularization::Real=0.0,
            row_subsample::Real=1.0, feature_subsample::Real=1.0)
        _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
            max_bins, min_samples_leaf, tolerance, l1_regularization,
            l2_regularization, row_subsample, feature_subsample)
        new(Int(n_estimators), Float64(learning_rate), Int(max_depth),
            Int(max_bins), Int(min_samples_leaf), Float64(tolerance),
            Float64(l1_regularization), Float64(l2_regularization),
            Float64(row_subsample), Float64(feature_subsample))
    end
end

function _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
                                       max_bins, min_samples_leaf, tolerance,
                                       l1_regularization, l2_regularization,
                                       row_subsample, feature_subsample)
    n_estimators > 0 || throw(InvalidHyperparameterError("n_estimators must be positive."))
    isfinite(learning_rate) && learning_rate > 0 || throw(InvalidHyperparameterError(
        "learning_rate must be finite and positive."))
    max_depth > 0 || throw(InvalidHyperparameterError("max_depth must be positive."))
    2 <= max_bins <= 255 || throw(InvalidHyperparameterError("max_bins must lie in 2:255."))
    min_samples_leaf > 0 || throw(InvalidHyperparameterError("min_samples_leaf must be positive."))
    isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
        "tolerance must be finite and nonnegative."))
    isfinite(l1_regularization) && l1_regularization >= 0 ||
        throw(InvalidHyperparameterError("l1_regularization must be finite and nonnegative."))
    isfinite(l2_regularization) && l2_regularization >= 0 ||
        throw(InvalidHyperparameterError("l2_regularization must be finite and nonnegative."))
    isfinite(row_subsample) && 0 < row_subsample <= 1 ||
        throw(InvalidHyperparameterError("row_subsample must lie in (0, 1]."))
    isfinite(feature_subsample) && 0 < feature_subsample <= 1 ||
        throw(InvalidHyperparameterError("feature_subsample must lie in (0, 1]."))
end

struct FittedHistGradientBoosting{M,T,L,TR,R,S} <: AbstractFittedEstimator
    model::M
    bin_edges::Vector{Vector{T}}
    trees::TR
    initial_prediction::Vector{T}
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:HistGradientBoostingRegressor}) = (task=:regression,
    sparse=false, missing=false, weights=true, partial_fit=false, probabilistic=false)
capabilities(::Type{<:HistGradientBoostingClassifier}) = (task=:classification,
    sparse=false, missing=false, weights=true, partial_fit=false, probabilistic=true)

function _fit_bin_edges(X, max_bins)
    T = eltype(X)
    edges = Vector{Vector{T}}(undef, size(X, 2))
    for feature in axes(X, 2)
        values = sort!(unique(view(X, :, feature)))
        if length(values) <= max_bins
            edges[feature] = T[(values[index] + values[index + 1]) / 2
                               for index in 1:length(values)-1]
        else
            positions = unique(clamp.(round.(Int,
                (1:max_bins-1) .* length(values) ./ max_bins), 1, length(values)-1))
            edges[feature] = T[(values[index] + values[index + 1]) / 2 for index in positions]
        end
    end
    edges
end

function _bin_data(X, edges, T)
    binned = Matrix{T}(undef, size(X))
    for feature in axes(X, 2), row in axes(X, 1)
        binned[row, feature] = T(searchsortedlast(edges[feature], X[row, feature]) + 1)
    end
    binned
end

function _boosting_weights(weights, n, T, name)
    result = weights === nothing ? ones(T, n) : T.(weights)
    length(result) == n || throw(SchemaMismatchError(
        "$name weights have length $(length(result)); expected $n."))
    all(weight -> isfinite(weight) && weight >= 0, result) && sum(result) > 0 ||
        throw(UnsupportedDataError("$name weights must be finite, nonnegative, and have positive sum."))
    result
end

function _boosting_sample_indices(model, observations, context, iteration, class)
    count = max(2, ceil(Int, model.row_subsample * observations))
    count >= observations && return collect(1:observations)
    rng = derive_context(context, :hist_gradient_boosting, :subsample,
                         iteration, class).rng
    sort!(randperm(rng, observations)[1:count])
end

function _tree_leaf_index(tree, row)
    node_index = 1
    while !tree.nodes[node_index].is_leaf
        node = tree.nodes[node_index]
        node_index = row[node.feature] <= node.threshold ? node.left : node.right
    end
    node_index
end

function _regularized_newton_tree(tree, X, gradients, hessians, weights,
                                  l1_regularization, l2_regularization)
    T = eltype(tree.feature_importances)
    numerators = zeros(T, length(tree.nodes))
    denominators = zeros(T, length(tree.nodes))
    for row in axes(X, 1)
        leaf = _tree_leaf_index(tree, view(X, row, :))
        numerators[leaf] += weights[row] * gradients[row]
        denominators[leaf] += weights[row] * hessians[row]
    end
    nodes = copy(tree.nodes)
    for index in eachindex(nodes)
        node = nodes[index]
        node.is_leaf || continue
        numerator = sign(numerators[index]) *
            max(abs(numerators[index]) - T(l1_regularization), zero(T))
        prediction = numerator / (denominators[index] + T(l2_regularization) + eps(T))
        nodes[index] = TreeNode(node.feature, node.threshold, node.left, node.right,
            prediction, node.predicted_class, node.probabilities, node.samples,
            node.weighted_samples, node.impurity, node.is_leaf)
    end
    FittedDecisionTree(tree.model, nodes, tree.classes, tree.feature_importances,
                       tree.report, tree.schema)
end

function fit(model::HistGradientBoostingRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "HistGradientBoostingRegressor fitting")
    _validate_regression_data(X, y, weights, "HistGradientBoostingRegressor")
    T = float(promote_type(eltype(X), eltype(y), weights === nothing ? eltype(y) : eltype(weights)))
    data, target = Matrix{T}(X), T.(y)
    observation_weights = _boosting_weights(weights, size(X, 1), T, "HistGradientBoostingRegressor")
    edges = _fit_bin_edges(data, model.max_bins)
    binned = _bin_data(data, edges, T)
    initial = sum(observation_weights .* target) / sum(observation_weights)
    predictions = fill(initial, length(target))
    trees = FittedDecisionTree[]
    history = T[]
    converged = false
    max_iterations = effective_max_iterations(context, model.n_estimators)
    tolerance = T(effective_tolerance(context, model.tolerance))
    for iteration in 1:max_iterations
        residual = target .- predictions
        tree_context = derive_context(context, :hist_gradient_boosting,
                                      :iteration, iteration, :class, 1)
        indices = _boosting_sample_indices(model, size(X, 1), context, iteration, 1)
        tree = fit(DecisionTreeRegressor(max_depth=model.max_depth,
            min_samples_leaf=min(model.min_samples_leaf, max(1, length(indices) ÷ 2)),
            max_features=model.feature_subsample),
            view(binned, indices, :), view(residual, indices);
            weights=view(observation_weights, indices), context=tree_context)
        tree = _regularized_newton_tree(tree, view(binned, indices, :),
            view(residual, indices), ones(T, length(indices)),
            view(observation_weights, indices), model.l1_regularization,
            model.l2_regularization)
        push!(trees, tree)
        predictions .+= T(model.learning_rate) .* predict(tree, binned)
        loss = sum(observation_weights .* abs2.(target .- predictions)) / sum(observation_weights)
        push!(history, loss)
        if length(history) > 1 && abs(history[end] - history[end - 1]) <= tolerance
            converged = true
            break
        end
    end
    details = (n_estimators=length(trees), converged=converged,
               objective_history=history, max_bins=model.max_bins,
               bins_per_feature=length.(edges), loss=:squared_error,
               second_order=true, l1_regularization=model.l1_regularization,
               l2_regularization=model.l2_regularization,
               row_subsample=model.row_subsample,
               feature_subsample=model.feature_subsample)
    FittedHistGradientBoosting(model, edges, trees, T[initial], nothing,
        FitReport(observations=size(X, 1), features=size(X, 2), backend=:cpu,
                  details=details, context=context), with_target(infer_schema(X), y))
end

function fit(model::HistGradientBoostingClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "HistGradientBoostingClassifier fitting")
    _validate_numeric_matrix(X, "HistGradientBoostingClassifier")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "HistGradientBoostingClassifier target has length $(length(y)); expected $(size(X, 1))."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "HistGradientBoostingClassifier requires observations and features."))
    classes = _classification_classes(y)
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = Matrix{T}(X)
    observation_weights = _boosting_weights(weights, size(X, 1), T, "HistGradientBoostingClassifier")
    edges = _fit_bin_edges(data, model.max_bins)
    binned = _bin_data(data, edges, T)
    trained_classes = length(classes) == 2 ? classes[end:end] : classes
    initial = Vector{T}(undef, length(trained_classes))
    logits = Matrix{T}(undef, size(X, 1), length(trained_classes))
    targets = Matrix{T}(undef, size(X, 1), length(trained_classes))
    for (column, class) in enumerate(trained_classes)
        targets[:, column] .= T.(y .== class)
        prior = clamp(sum(observation_weights .* view(targets, :, column)) /
                      sum(observation_weights), eps(T), one(T) - eps(T))
        initial[column] = log(prior / (one(T) - prior))
        logits[:, column] .= initial[column]
    end
    trees = [FittedDecisionTree[] for _ in eachindex(trained_classes)]
    history = T[]
    converged = false
    max_iterations = effective_max_iterations(context, model.n_estimators)
    tolerance = T(effective_tolerance(context, model.tolerance))
    for iteration in 1:max_iterations
        for column in eachindex(trained_classes)
            probabilities = Kernels.sigmoid(view(logits, :, column))
            residual = view(targets, :, column) .- probabilities
            hessians = max.(probabilities .* (one(T) .- probabilities), eps(T))
            tree_context = derive_context(context, :hist_gradient_boosting,
                                          :iteration, iteration, :class, column)
            indices = _boosting_sample_indices(
                model, size(X, 1), context, iteration, column)
            newton_targets = residual[indices] ./ hessians[indices]
            newton_weights = observation_weights[indices] .* hessians[indices]
            tree = fit(DecisionTreeRegressor(max_depth=model.max_depth,
                min_samples_leaf=min(model.min_samples_leaf, max(1, length(indices) ÷ 2)),
                max_features=model.feature_subsample),
                view(binned, indices, :), newton_targets;
                weights=newton_weights, context=tree_context)
            tree = _regularized_newton_tree(tree, view(binned, indices, :),
                view(residual, indices), view(hessians, indices),
                view(observation_weights, indices), model.l1_regularization,
                model.l2_regularization)
            push!(trees[column], tree)
            logits[:, column] .+= T(model.learning_rate) .* predict(tree, binned)
        end
        loss = zero(T)
        for column in eachindex(trained_classes)
            column_logits = view(logits, :, column)
            column_targets = view(targets, :, column)
            losses = max.(column_logits, zero(T)) .- column_logits .* column_targets .+
                     log1p.(exp.(-abs.(column_logits)))
            loss += sum(observation_weights .* losses) / sum(observation_weights)
        end
        loss /= length(trained_classes)
        push!(history, loss)
        if length(history) > 1 && abs(history[end] - history[end - 1]) <= tolerance
            converged = true
            break
        end
    end
    details = (n_estimators=length(first(trees)), converged=converged,
               objective_history=history, max_bins=model.max_bins,
               bins_per_feature=length.(edges), loss=:log_loss,
               class_order=copy(classes), strategy=:one_vs_rest,
               second_order=true, l1_regularization=model.l1_regularization,
               l2_regularization=model.l2_regularization,
               row_subsample=model.row_subsample,
               feature_subsample=model.feature_subsample)
    schema = with_class_target(infer_schema(X), classes)
    FittedHistGradientBoosting(model, edges, trees, initial, classes,
        FitReport(observations=size(X, 1), features=size(X, 2), backend=:cpu,
                  details=details, context=context), schema)
end

function predict(fitted::FittedHistGradientBoosting{<:HistGradientBoostingRegressor},
                 X::AbstractMatrix)
    _validate_numeric_matrix(X, "HistGradientBoostingRegressor")
    _validate_feature_count(fitted.schema, X, "HistGradientBoostingRegressor")
    binned = _bin_data(X, fitted.bin_edges, eltype(fitted.initial_prediction))
    result = fill(first(fitted.initial_prediction), size(X, 1))
    for tree in fitted.trees
        result .+= eltype(result)(fitted.model.learning_rate) .* predict(tree, binned)
    end
    result
end

function predict_proba(fitted::FittedHistGradientBoosting{<:HistGradientBoostingClassifier},
                       X::AbstractMatrix)
    _validate_numeric_matrix(X, "HistGradientBoostingClassifier")
    _validate_feature_count(fitted.schema, X, "HistGradientBoostingClassifier")
    T = eltype(fitted.initial_prediction)
    binned = _bin_data(X, fitted.bin_edges, T)
    logits = repeat(transpose(fitted.initial_prediction), size(X, 1), 1)
    for column in eachindex(fitted.trees), tree in fitted.trees[column]
        logits[:, column] .+= T(fitted.model.learning_rate) .* predict(tree, binned)
    end
    positive = Kernels.sigmoid(logits)
    length(fitted.classes) == 2 && return hcat(one(T) .- vec(positive), vec(positive))
    positive ./ sum(positive; dims=2)
end

function predict(fitted::FittedHistGradientBoosting{<:HistGradientBoostingClassifier},
                 X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

report(fitted::FittedHistGradientBoosting) = fitted.report
