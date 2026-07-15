abstract type AbstractDecisionTree <: AbstractPredictor end

struct DecisionTreeClassifier <: AbstractDecisionTree
    criterion::Symbol
    max_depth::Union{Nothing,Int}
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Union{Nothing,Int,Float64,Symbol}
    min_impurity_decrease::Float64
    splitter::Symbol
    function DecisionTreeClassifier(; criterion::Symbol=:gini, max_depth=nothing,
            min_samples_split::Integer=2, min_samples_leaf::Integer=1,
            max_features=nothing, min_impurity_decrease::Real=0.0,
            splitter::Symbol=:best)
        criterion in (:gini, :entropy) || throw(InvalidHyperparameterError(
            "DecisionTreeClassifier criterion must be :gini or :entropy."))
        _validate_tree_parameters(max_depth, min_samples_split, min_samples_leaf,
                                  max_features, min_impurity_decrease)
        splitter in (:best, :random) || throw(InvalidHyperparameterError(
            "DecisionTreeClassifier splitter must be :best or :random."))
        new(criterion, max_depth, Int(min_samples_split), Int(min_samples_leaf),
            max_features, Float64(min_impurity_decrease), splitter)
    end
end

struct DecisionTreeRegressor <: AbstractDecisionTree
    criterion::Symbol
    max_depth::Union{Nothing,Int}
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Union{Nothing,Int,Float64,Symbol}
    min_impurity_decrease::Float64
    splitter::Symbol
    function DecisionTreeRegressor(; criterion::Symbol=:squared_error, max_depth=nothing,
            min_samples_split::Integer=2, min_samples_leaf::Integer=1,
            max_features=nothing, min_impurity_decrease::Real=0.0,
            splitter::Symbol=:best)
        criterion === :squared_error || throw(InvalidHyperparameterError(
            "DecisionTreeRegressor criterion must be :squared_error."))
        _validate_tree_parameters(max_depth, min_samples_split, min_samples_leaf,
                                  max_features, min_impurity_decrease)
        splitter in (:best, :random) || throw(InvalidHyperparameterError(
            "DecisionTreeRegressor splitter must be :best or :random."))
        new(criterion, max_depth, Int(min_samples_split), Int(min_samples_leaf),
            max_features, Float64(min_impurity_decrease), splitter)
    end
end

function _validate_tree_parameters(max_depth, min_samples_split, min_samples_leaf,
                                   max_features, min_impurity_decrease)
    max_depth === nothing || (max_depth isa Integer && max_depth > 0) ||
        throw(InvalidHyperparameterError("max_depth must be positive or nothing."))
    min_samples_split >= 2 || throw(InvalidHyperparameterError("min_samples_split must be at least 2."))
    min_samples_leaf >= 1 || throw(InvalidHyperparameterError("min_samples_leaf must be positive."))
    valid_features = max_features === nothing ||
        (max_features isa Integer && max_features > 0) ||
        (max_features isa AbstractFloat && 0 < max_features <= 1) ||
        max_features in (:sqrt, :log2)
    valid_features || throw(InvalidHyperparameterError(
        "max_features must be nothing, a positive integer, a fraction in (0,1], :sqrt, or :log2."))
    isfinite(min_impurity_decrease) && min_impurity_decrease >= 0 ||
        throw(InvalidHyperparameterError("min_impurity_decrease must be finite and nonnegative."))
end

struct TreeNode{T}
    feature::Int
    threshold::T
    left::Int
    right::Int
    prediction::T
    predicted_class::Int
    probabilities::Vector{T}
    samples::Int
    weighted_samples::T
    impurity::T
    is_leaf::Bool
end

struct FittedDecisionTree{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    nodes::Vector{TreeNode{T}}
    classes::L
    feature_importances::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:DecisionTreeClassifier}) = (task=:classification, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=true)
capabilities(::Type{<:DecisionTreeRegressor}) = (task=:regression, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=false)

function _tree_feature_count(max_features, p)
    max_features === nothing && return p
    max_features === :sqrt && return max(1, floor(Int, sqrt(p)))
    max_features === :log2 && return max(1, floor(Int, log2(p)))
    max_features isa Integer && return min(Int(max_features), p)
    max(1, min(p, floor(Int, max_features * p)))
end

function _tree_features(model, p, rng)
    count = _tree_feature_count(model.max_features, p)
    count == p ? collect(1:p) : sort!(randperm(rng, p)[1:count])
end

function _class_impurity(encoded, weights, indices, n_classes, criterion, T)
    total = sum(view(weights, indices))
    counts = zeros(T, n_classes)
    for index in indices
        counts[encoded[index]] += weights[index]
    end
    probabilities = counts ./ total
    impurity = criterion === :gini ? one(T) - sum(abs2, probabilities) :
        -sum(probability -> iszero(probability) ? zero(T) : probability * log2(probability), probabilities)
    impurity, probabilities
end

function _regression_impurity(target, weights, indices, T)
    local_weights = view(weights, indices)
    total = sum(local_weights)
    average = sum(local_weights .* view(target, indices)) / total
    variance = sum(local_weights .* abs2.(view(target, indices) .- average)) / total
    variance, average
end

function _node_summary(classifier, target, weights, indices, classes, criterion, T)
    if classifier
        impurity, probabilities = _class_impurity(target, weights, indices, length(classes), criterion, T)
        prediction_class = argmax(probabilities)
        return impurity, zero(T), prediction_class, probabilities
    end
    impurity, prediction = _regression_impurity(target, weights, indices, T)
    impurity, prediction, 0, T[]
end

@inline function _tree_sum_abs2(values)
    total = zero(eltype(values))
    @inbounds @simd for value in values
        total += abs2(value)
    end
    total
end

function _best_binary_gini_split(model, X, target, indices, parent_impurity, rng, T)
    best_gain = -T(Inf)
    best_feature = 0
    best_threshold = zero(T)
    best_ordered = Int[]
    best_position = 0
    node_count = length(indices)
    total_first_class = count(index -> target[index] == 1, indices)
    for feature in _tree_features(model, size(X, 2), rng)
        ordered = sort(indices; by=index -> (X[index, feature], index))
        first_value = X[first(ordered), feature]
        last_value = X[last(ordered), feature]
        first_value == last_value && continue
        random_threshold = model.splitter === :random ?
            first_value + rand(rng, T) * (last_value - first_value) : nothing
        random_position = 0
        if random_threshold !== nothing
            @inbounds for position in eachindex(ordered)
                X[ordered[position], feature] <= random_threshold || break
                random_position = position
            end
        end
        left_first_class = 0
        @inbounds for position in 1:node_count-1
            left_first_class += target[ordered[position]] == 1
            position < model.min_samples_leaf && continue
            node_count - position < model.min_samples_leaf && break
            model.splitter === :random && position != random_position && continue
            left_value = X[ordered[position], feature]
            right_value = X[ordered[position + 1], feature]
            left_value == right_value && continue
            left_second_class = position - left_first_class
            right_first_class = total_first_class - left_first_class
            right_count = node_count - position
            right_second_class = right_count - right_first_class
            left_impurity = one(T) -
                (abs2(T(left_first_class)) + abs2(T(left_second_class))) /
                abs2(T(position))
            right_impurity = one(T) -
                (abs2(T(right_first_class)) + abs2(T(right_second_class))) /
                abs2(T(right_count))
            gain = parent_impurity -
                (T(position) * left_impurity + T(right_count) * right_impurity) /
                T(node_count)
            if gain > best_gain
                best_gain = gain
                best_feature = feature
                best_threshold = model.splitter === :best ?
                    T(left_value / 2 + right_value / 2) : T(random_threshold)
                best_ordered = ordered
                best_position = position
            end
        end
    end
    best_left = best_position == 0 ? Int[] : best_ordered[1:best_position]
    best_right = best_position == 0 ? Int[] : best_ordered[best_position + 1:end]
    best_gain, best_feature, best_threshold, best_left, best_right
end

function _best_tree_split(model, X, target, weights, indices, classes,
                          parent_impurity, rng, T, unit_weights)
    if unit_weights && model isa DecisionTreeClassifier &&
       model.criterion === :gini && length(classes) == 2
        return _best_binary_gini_split(
            model, X, target, indices, parent_impurity, rng, T)
    end
    best_gain = -T(Inf)
    best_feature = 0
    best_threshold = zero(T)
    best_ordered = Int[]
    best_position = 0
    classifier = model isa DecisionTreeClassifier
    total_weight = sum(view(weights, indices))
    for feature in _tree_features(model, size(X, 2), rng)
        ordered = sort(indices; by=index -> (X[index, feature], index))
        first_value, last_value = X[first(ordered), feature], X[last(ordered), feature]
        first_value == last_value && continue
        random_threshold = model.splitter === :random ?
            first_value + rand(rng, T) * (last_value - first_value) : nothing
        random_position = random_threshold === nothing ? 0 :
            searchsortedlast([X[index, feature] for index in ordered], random_threshold)

        left_weight = zero(T)
        if classifier
            left_counts = zeros(T, length(classes))
            right_counts = zeros(T, length(classes))
            for index in ordered
                right_counts[target[index]] += weights[index]
            end
        else
            left_sum = zero(T)
            left_squared_sum = zero(T)
            right_sum = sum(index -> weights[index] * target[index], ordered)
            right_squared_sum = sum(index -> weights[index] * abs2(target[index]), ordered)
        end

        @inbounds for position in 1:length(ordered)-1
            index = ordered[position]
            weight = weights[index]
            left_weight += weight
            if classifier
                class = target[index]
                left_counts[class] += weight
                right_counts[class] -= weight
            else
                contribution = weight * target[index]
                squared_contribution = weight * abs2(target[index])
                left_sum += contribution
                left_squared_sum += squared_contribution
                right_sum -= contribution
                right_squared_sum -= squared_contribution
            end

            position < model.min_samples_leaf && continue
            length(ordered) - position < model.min_samples_leaf && break
            model.splitter === :random && position != random_position && continue
            left_value = X[ordered[position], feature]
            right_value = X[ordered[position + 1], feature]
            left_value == right_value && continue
            right_weight = total_weight - left_weight
            (left_weight <= zero(T) || right_weight <= zero(T)) && continue
            if classifier
                if model.criterion === :gini
                    left_impurity = one(T) - _tree_sum_abs2(left_counts) / abs2(left_weight)
                    right_impurity = one(T) - _tree_sum_abs2(right_counts) / abs2(right_weight)
                else
                    left_impurity = -sum(count -> iszero(count) ? zero(T) :
                        (count / left_weight) * log2(count / left_weight), left_counts)
                    right_impurity = -sum(count -> iszero(count) ? zero(T) :
                        (count / right_weight) * log2(count / right_weight), right_counts)
                end
            else
                left_impurity = max(left_squared_sum / left_weight -
                    abs2(left_sum / left_weight), zero(T))
                right_impurity = max(right_squared_sum / right_weight -
                    abs2(right_sum / right_weight), zero(T))
            end
            gain = parent_impurity - (left_weight * left_impurity +
                right_weight * right_impurity) / total_weight
            if gain > best_gain
                best_gain = gain
                best_feature = feature
                best_threshold = model.splitter === :best ? T(left_value / 2 + right_value / 2) :
                    T(random_threshold)
                best_ordered = ordered
                best_position = position
            end
        end
    end
    best_left = best_position == 0 ? Int[] : best_ordered[1:best_position]
    best_right = best_position == 0 ? Int[] : best_ordered[best_position + 1:end]
    best_gain, best_feature, best_threshold, best_left, best_right
end

function _build_tree!(nodes, importances, model, X, target, weights, indices,
                      classes, depth, rng, T, unit_weights)
    classifier = model isa DecisionTreeClassifier
    impurity, prediction, predicted_class, probabilities = _node_summary(
        classifier, target, weights, indices, classes, model.criterion, T)
    node_index = length(nodes) + 1
    push!(nodes, TreeNode(0, zero(T), 0, 0, prediction, predicted_class,
                          probabilities, length(indices), sum(view(weights, indices)),
                          impurity, true))
    depth_limit = model.max_depth !== nothing && depth >= model.max_depth
    pure = impurity <= eps(T)
    splittable = length(indices) >= model.min_samples_split &&
                 length(indices) >= 2 * model.min_samples_leaf
    (depth_limit || pure || !splittable) && return node_index, depth
    gain, feature, threshold, left_indices, right_indices = _best_tree_split(
        model, X, target, weights, indices, classes, impurity, rng, T, unit_weights)
    (feature == 0 || gain < model.min_impurity_decrease || gain <= eps(T)) &&
        return node_index, depth
    left, left_depth = _build_tree!(nodes, importances, model, X, target, weights,
        left_indices, classes, depth + 1, rng, T, unit_weights)
    right, right_depth = _build_tree!(nodes, importances, model, X, target, weights,
        right_indices, classes, depth + 1, rng, T, unit_weights)
    importances[feature] += gain * sum(view(weights, indices))
    nodes[node_index] = TreeNode(feature, threshold, left, right, prediction,
        predicted_class, probabilities, length(indices), sum(view(weights, indices)), impurity, false)
    node_index, max(left_depth, right_depth)
end

function _fit_decision_tree(model, X, target, classes, weights, context)
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_numeric_matrix(X, name)
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError("$name requires observations and features."))
    length(target) == n || throw(SchemaMismatchError(
        "$name target has length $(length(target)); expected $n."))
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = Matrix{T}(X)
    observation_weights = weights === nothing ? ones(T, n) : T.(weights)
    length(observation_weights) == n || throw(SchemaMismatchError(
        "$name weights have length $(length(observation_weights)); expected $n."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) &&
        sum(observation_weights) > 0 || throw(UnsupportedDataError(
        "$name weights must be finite, nonnegative, and have positive sum."))
    nodes = TreeNode{T}[]
    importances = zeros(T, p)
    _, maximum_depth = _build_tree!(nodes, importances, model, data, target,
        observation_weights, collect(1:n), classes, 0, context.rng, T,
        weights === nothing)
    total_importance = sum(importances)
    iszero(total_importance) || (importances ./= total_importance)
    leaves = count(node -> node.is_leaf, nodes)
    details = (nodes=length(nodes), leaves=leaves, maximum_depth=maximum_depth,
               criterion=model.criterion, feature_importances=copy(importances))
    schema = infer_schema(X)
    schema = classes === nothing ? with_target(schema, target) :
             with_class_target(schema, classes)
    FittedDecisionTree(model, nodes, classes, importances,
        FitReport(observations=n, features=p, backend=:cpu, details=details,
                  context=context), schema)
end

function fit(model::DecisionTreeClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    encoded = [searchsortedfirst(classes, value) for value in y]
    _fit_decision_tree(model, X, encoded, classes, weights, context)
end

function fit(model::DecisionTreeRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    eltype(y) <: Number && all(isfinite, y) || throw(UnsupportedDataError(
        "DecisionTreeRegressor requires a finite numeric target."))
    _fit_decision_tree(model, X, y, nothing, weights, context)
end

function _tree_leaf(fitted, row)
    node_index = 1
    while !fitted.nodes[node_index].is_leaf
        node = fitted.nodes[node_index]
        node_index = row[node.feature] <= node.threshold ? node.left : node.right
    end
    fitted.nodes[node_index]
end

function predict(fitted::FittedDecisionTree{<:DecisionTreeRegressor}, X::AbstractMatrix)
    _validate_numeric_matrix(X, "DecisionTreeRegressor")
    _validate_feature_count(fitted.schema, X, "DecisionTreeRegressor")
    [_tree_leaf(fitted, view(X, row, :)).prediction for row in axes(X, 1)]
end

function predict_proba(fitted::FittedDecisionTree{<:DecisionTreeClassifier}, X::AbstractMatrix)
    _validate_numeric_matrix(X, "DecisionTreeClassifier")
    _validate_feature_count(fitted.schema, X, "DecisionTreeClassifier")
    probabilities = Matrix{eltype(fitted.feature_importances)}(undef, size(X, 1), length(fitted.classes))
    for row in axes(X, 1)
        probabilities[row, :] .= _tree_leaf(fitted, view(X, row, :)).probabilities
    end
    probabilities
end

function predict(fitted::FittedDecisionTree{<:DecisionTreeClassifier}, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

report(fitted::FittedDecisionTree) = fitted.report
