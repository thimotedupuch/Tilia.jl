abstract type AbstractForest <: AbstractPredictor end

for (name, task) in ((:RandomForestClassifier, :classification),
                     (:RandomForestRegressor, :regression),
                     (:ExtraTreesClassifier, :classification),
                     (:ExtraTreesRegressor, :regression))
    @eval begin
        struct $name <: AbstractForest
            n_estimators::Int
            max_depth::Union{Nothing,Int}
            min_samples_split::Int
            min_samples_leaf::Int
            max_features::Union{Nothing,Int,Float64,Symbol}
            bootstrap::Bool
            function $name(; n_estimators::Integer=100, max_depth=nothing,
                    min_samples_split::Integer=2, min_samples_leaf::Integer=1,
                    max_features=$(task === :classification ? QuoteNode(:sqrt) : :(nothing)),
                    bootstrap::Bool=$(startswith(string(name), "RandomForest")))
                n_estimators > 0 || throw(InvalidHyperparameterError("$($name) n_estimators must be positive."))
                _validate_tree_parameters(max_depth, min_samples_split, min_samples_leaf,
                                           max_features, 0.0)
                new(Int(n_estimators), max_depth, Int(min_samples_split),
                    Int(min_samples_leaf), max_features, bootstrap)
            end
        end
    end
end

struct FittedForest{M,T,L,TR,R,S} <: AbstractFittedEstimator
    model::M
    trees::Vector{TR}
    classes::L
    feature_importances::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:Union{RandomForestClassifier,ExtraTreesClassifier}}) =
    (task=:classification, sparse=false, missing=false, weights=true,
     partial_fit=false, probabilistic=true)
capabilities(::Type{<:Union{RandomForestRegressor,ExtraTreesRegressor}}) =
    (task=:regression, sparse=false, missing=false, weights=true,
     partial_fit=false, probabilistic=false)

_forest_classifier(model) = model isa Union{RandomForestClassifier,ExtraTreesClassifier}
_forest_extra(model) = model isa Union{ExtraTreesClassifier,ExtraTreesRegressor}

function _forest_tree(model)
    common = (max_depth=model.max_depth, min_samples_split=model.min_samples_split,
              min_samples_leaf=model.min_samples_leaf, max_features=model.max_features,
              splitter=_forest_extra(model) ? :random : :best)
    _forest_classifier(model) ? DecisionTreeClassifier(; common...) :
                                DecisionTreeRegressor(; common...)
end

function _fit_forest(model, X, target, classes, weights, context)
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_numeric_matrix(X, name)
    n, p = size(X)
    length(target) == n || throw(SchemaMismatchError(
        "$name target has length $(length(target)); expected $n."))
    n > 0 && p > 0 || throw(UnsupportedDataError("$name requires observations and features."))
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = Matrix{T}(X)
    observation_weights = weights === nothing ? ones(T, n) : T.(weights)
    length(observation_weights) == n || throw(SchemaMismatchError(
        "$name weights have length $(length(observation_weights)); expected $n."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) &&
        sum(observation_weights) > 0 || throw(UnsupportedDataError(
        "$name weights must be finite, nonnegative, and have positive sum."))
    tree_model = _forest_tree(model)
    trees = FittedDecisionTree[]
    for tree_index in 1:model.n_estimators
        tree_context = derive_context(context, :forest, :tree, tree_index)
        indices = model.bootstrap ? rand(tree_context.rng, 1:n, n) : collect(1:n)
        push!(trees, _fit_decision_tree(tree_model, view(data, indices, :),
            target[indices], classes, observation_weights[indices], tree_context))
    end
    importances = vec(mean(reduce(hcat, [tree.feature_importances for tree in trees]); dims=2))
    details = (n_estimators=model.n_estimators, bootstrap=model.bootstrap,
               splitter=_forest_extra(model) ? :random : :best,
               mean_nodes=mean(length(tree.nodes) for tree in trees),
               feature_importances=copy(importances))
    schema = infer_schema(X)
    schema = classes === nothing ? with_target(schema, target) :
             with_class_target(schema, classes)
    FittedForest(model, trees, classes, importances,
        FitReport(observations=n, features=p, backend=:cpu, details=details,
                  context=context), schema)
end

function fit(model::Union{RandomForestClassifier,ExtraTreesClassifier},
             X::AbstractMatrix, y::AbstractVector; weights=nothing, context=default_context())
    classes = _classification_classes(y)
    encoded = [searchsortedfirst(classes, value) for value in y]
    _fit_forest(model, X, encoded, classes, weights, context)
end

function fit(model::Union{RandomForestRegressor,ExtraTreesRegressor},
             X::AbstractMatrix, y::AbstractVector; weights=nothing, context=default_context())
    eltype(y) <: Number && all(isfinite, y) || throw(UnsupportedDataError(
        "$(nameof(typeof(model))) requires a finite numeric target."))
    _fit_forest(model, X, y, nothing, weights, context)
end

function predict_proba(fitted::FittedForest{<:Union{RandomForestClassifier,ExtraTreesClassifier}},
                       X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    result = zeros(eltype(fitted.feature_importances), size(X, 1), length(fitted.classes))
    for tree in fitted.trees
        result .+= predict_proba(tree, X)
    end
    result ./ length(fitted.trees)
end

function predict(fitted::FittedForest{<:Union{RandomForestClassifier,ExtraTreesClassifier}},
                 X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

function predict(fitted::FittedForest{<:Union{RandomForestRegressor,ExtraTreesRegressor}},
                 X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    predictions = zeros(eltype(fitted.feature_importances), size(X, 1))
    for tree in fitted.trees
        predictions .+= predict(tree, X)
    end
    predictions ./ length(fitted.trees)
end

report(fitted::FittedForest) = fitted.report
