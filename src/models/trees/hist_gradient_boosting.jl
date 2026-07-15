abstract type AbstractHistGradientBoosting <: AbstractPredictor end

struct HistGradientBoostingRegressor <: AbstractHistGradientBoosting
    n_estimators::Int
    learning_rate::Float64
    max_depth::Int
    max_bins::Int
    min_samples_leaf::Int
    tolerance::Float64
    function HistGradientBoostingRegressor(; n_estimators::Integer=100,
            learning_rate::Real=0.1, max_depth::Integer=3, max_bins::Integer=255,
            min_samples_leaf::Integer=5, tolerance::Real=1e-7)
        _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
            max_bins, min_samples_leaf, tolerance)
        new(Int(n_estimators), Float64(learning_rate), Int(max_depth),
            Int(max_bins), Int(min_samples_leaf), Float64(tolerance))
    end
end

struct HistGradientBoostingClassifier <: AbstractHistGradientBoosting
    n_estimators::Int
    learning_rate::Float64
    max_depth::Int
    max_bins::Int
    min_samples_leaf::Int
    tolerance::Float64
    function HistGradientBoostingClassifier(; n_estimators::Integer=100,
            learning_rate::Real=0.1, max_depth::Integer=3, max_bins::Integer=255,
            min_samples_leaf::Integer=5, tolerance::Real=1e-7)
        _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
            max_bins, min_samples_leaf, tolerance)
        new(Int(n_estimators), Float64(learning_rate), Int(max_depth),
            Int(max_bins), Int(min_samples_leaf), Float64(tolerance))
    end
end

function _validate_boosting_parameters(n_estimators, learning_rate, max_depth,
                                       max_bins, min_samples_leaf, tolerance)
    n_estimators > 0 || throw(InvalidHyperparameterError("n_estimators must be positive."))
    isfinite(learning_rate) && learning_rate > 0 || throw(InvalidHyperparameterError(
        "learning_rate must be finite and positive."))
    max_depth > 0 || throw(InvalidHyperparameterError("max_depth must be positive."))
    2 <= max_bins <= 255 || throw(InvalidHyperparameterError("max_bins must lie in 2:255."))
    min_samples_leaf > 0 || throw(InvalidHyperparameterError("min_samples_leaf must be positive."))
    isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
        "tolerance must be finite and nonnegative."))
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
        tree = fit(DecisionTreeRegressor(max_depth=model.max_depth,
            min_samples_leaf=min(model.min_samples_leaf, max(1, size(X, 1) ÷ 2))),
            binned, residual; weights=observation_weights, context=tree_context)
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
               bins_per_feature=length.(edges), loss=:squared_error)
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
            residual = view(targets, :, column) .- Kernels.sigmoid(view(logits, :, column))
            tree_context = derive_context(context, :hist_gradient_boosting,
                                          :iteration, iteration, :class, column)
            tree = fit(DecisionTreeRegressor(max_depth=model.max_depth,
                min_samples_leaf=min(model.min_samples_leaf, max(1, size(X, 1) ÷ 2))),
                binned, residual; weights=observation_weights, context=tree_context)
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
               class_order=copy(classes), strategy=:one_vs_rest)
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
