abstract type AbstractSGDLinearModel <: AbstractPredictor end

"""Mini-batch stochastic-gradient linear regression with incremental updates."""
struct SGDRegressor <: AbstractSGDLinearModel
    learning_rate::Float64
    l2::Float64
    epochs::Int
    batch_size::Int
    fit_intercept::Bool
    shuffle::Bool
    function SGDRegressor(; learning_rate::Real=0.01, l2::Real=0.0001,
                          epochs::Integer=20, batch_size::Integer=32,
                          fit_intercept::Bool=true, shuffle::Bool=true)
        _validate_sgd_parameters(learning_rate, l2, epochs, batch_size)
        new(Float64(learning_rate), Float64(l2), Int(epochs), Int(batch_size),
            fit_intercept, shuffle)
    end
end

"""Mini-batch stochastic-gradient softmax classifier with incremental updates."""
struct SGDClassifier <: AbstractSGDLinearModel
    learning_rate::Float64
    l2::Float64
    epochs::Int
    batch_size::Int
    fit_intercept::Bool
    shuffle::Bool
    function SGDClassifier(; learning_rate::Real=0.01, l2::Real=0.0001,
                           epochs::Integer=20, batch_size::Integer=32,
                           fit_intercept::Bool=true, shuffle::Bool=true)
        _validate_sgd_parameters(learning_rate, l2, epochs, batch_size)
        new(Float64(learning_rate), Float64(l2), Int(epochs), Int(batch_size),
            fit_intercept, shuffle)
    end
end

function _validate_sgd_parameters(learning_rate, l2, epochs, batch_size)
    isfinite(learning_rate) && learning_rate > 0 || throw(InvalidHyperparameterError(
        "SGD learning_rate must be finite and positive."))
    isfinite(l2) && l2 >= 0 || throw(InvalidHyperparameterError(
        "SGD l2 must be finite and nonnegative."))
    epochs > 0 || throw(InvalidHyperparameterError("SGD epochs must be positive."))
    batch_size > 0 || throw(InvalidHyperparameterError("SGD batch_size must be positive."))
end

struct FittedSGDRegressor{M,T,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::Vector{T}
    intercept::T
    updates::Int
    report::R
    schema::S
end

struct FittedSGDClassifier{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::Matrix{T}
    intercept::Vector{T}
    classes::Vector{L}
    updates::Int
    report::R
    schema::S
end

capabilities(::Type{<:SGDRegressor}) = (task=:regression, sparse=true,
    missing=false, weights=false, partial_fit=true, probabilistic=false)
capabilities(::Type{<:SGDClassifier}) = (task=:classification, sparse=true,
    missing=false, weights=false, partial_fit=true, probabilistic=true)

function _sgd_batches(model, n, context, epoch)
    ordering = model.shuffle ? randperm(
        derive_context(context, :sgd, :epoch, epoch).rng, n) : collect(1:n)
    (view(ordering, first:min(first + model.batch_size - 1, n))
     for first in 1:model.batch_size:n)
end

_sgd_step(model, updates, T) = T(model.learning_rate) / sqrt(T(updates + 1))

function _train_sgd_regressor!(model, coefficients, intercept, updates,
                               X, y, epochs, context)
    T = eltype(coefficients)
    history = T[]
    for epoch in 1:epochs
        for indices in _sgd_batches(model, size(X, 1), context, epoch)
            batch = view(X, indices, :)
            residual = batch * coefficients .+ intercept .- view(y, indices)
            inverse_count = inv(T(length(indices)))
            gradient = vec(transpose(batch) * residual) .* inverse_count .+
                T(model.l2) .* coefficients
            intercept_gradient = model.fit_intercept ? sum(residual) * inverse_count : zero(T)
            step = _sgd_step(model, updates, T)
            coefficients .-= step .* gradient
            intercept -= step * intercept_gradient
            updates += 1
        end
        residual = X * coefficients .+ intercept .- y
        push!(history, sum(abs2, residual) / (T(2) * length(y)) +
              T(model.l2) * sum(abs2, coefficients) / T(2))
    end
    intercept, updates, history
end

function fit(model::SGDRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "SGDRegressor fitting")
    _validate_regression_data(X, y, nothing, "SGDRegressor")
    T = float(promote_type(eltype(X), eltype(y)))
    data, target = T.(X), T.(y)
    coefficients = zeros(T, size(X, 2))
    intercept, updates, history = _train_sgd_regressor!(model, coefficients,
        zero(T), 0, data, target, model.epochs, context)
    details = (solver=:mini_batch_sgd, epochs=model.epochs, updates,
               objective_history=history, l2=T(model.l2), batch_size=model.batch_size)
    FittedSGDRegressor(model, coefficients, intercept, updates,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), with_target(infer_schema(X), y))
end

partial_fit(model::SGDRegressor, X::AbstractMatrix, y::AbstractVector; kwargs...) =
    _partial_fit_new_sgd_regressor(model, X, y; kwargs...)

function _partial_fit_new_sgd_regressor(model, X, y; weights=nothing,
                                        context=default_context())
    one_epoch = SGDRegressor(learning_rate=model.learning_rate, l2=model.l2,
        epochs=1, batch_size=model.batch_size, fit_intercept=model.fit_intercept,
        shuffle=model.shuffle)
    fitted = fit(one_epoch, X, y; weights, context)
    FittedSGDRegressor(model, fitted.coefficients, fitted.intercept, fitted.updates,
        fitted.report, fitted.schema)
end

function partial_fit(fitted::FittedSGDRegressor, X::AbstractMatrix, y::AbstractVector;
                     weights=nothing, context=default_context())
    reject_unsupported_weights(fitted.model, weights)
    _validate_regression_data(X, y, nothing, "SGDRegressor")
    _validate_feature_count(fitted.schema, X, "SGDRegressor")
    T = eltype(fitted.coefficients)
    coefficients = copy(fitted.coefficients)
    intercept, updates, history = _train_sgd_regressor!(fitted.model, coefficients,
        fitted.intercept, fitted.updates, T.(X), T.(y), 1, context)
    details = merge(fitted.report.details,
        (updates=updates, objective_history=vcat(fitted.report.details.objective_history, history),))
    FittedSGDRegressor(fitted.model, coefficients, intercept, updates,
        FitReport(observations=fitted.report.observations + size(X, 1),
                  features=size(X, 2), details=details, context=context), fitted.schema)
end

function _train_sgd_classifier!(model, coefficients, intercept, updates,
                                X, targets, epochs, context)
    T = eltype(coefficients)
    history = T[]
    for epoch in 1:epochs
        for indices in _sgd_batches(model, size(X, 1), context, epoch)
            batch = view(X, indices, :)
            probabilities = Kernels.softmax(
                batch * transpose(coefficients) .+ transpose(intercept); dims=2)
            residual = probabilities
            @inbounds for (row, source_index) in enumerate(indices)
                residual[row, targets[source_index]] -= one(T)
            end
            inverse_count = inv(T(length(indices)))
            gradient = Matrix(transpose(residual) * batch) .* inverse_count .+
                T(model.l2) .* coefficients
            intercept_gradient = model.fit_intercept ?
                vec(sum(residual; dims=1)) .* inverse_count : zeros(T, length(intercept))
            step = _sgd_step(model, updates, T)
            coefficients .-= step .* gradient
            intercept .-= step .* intercept_gradient
            updates += 1
        end
        probabilities = Kernels.softmax(
            X * transpose(coefficients) .+ transpose(intercept); dims=2)
        loss = -sum(log(max(probabilities[row, targets[row]], eps(T)))
                    for row in eachindex(targets)) / length(targets)
        push!(history, loss + T(model.l2) * sum(abs2, coefficients) / T(2))
    end
    updates, history
end

function _sgd_class_targets(y, classes)
    lookup = Dict(class => index for (index, class) in enumerate(classes))
    all(haskey(lookup, value) for value in y) || throw(SchemaMismatchError(
        "SGDClassifier partial_fit received a class absent from the initial batch."))
    [lookup[value] for value in y]
end

function fit(model::SGDClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "SGDClassifier fitting")
    _validate_numeric_matrix(X, "SGDClassifier")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "SGDClassifier target must match the observation count."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "SGDClassifier requires observations and features."))
    classes = _classification_classes(y)
    T = float(eltype(X))
    data = T.(X)
    coefficients = zeros(T, length(classes), size(X, 2))
    intercept = zeros(T, length(classes))
    targets = _sgd_class_targets(y, classes)
    updates, history = _train_sgd_classifier!(model, coefficients, intercept,
        0, data, targets, model.epochs, context)
    details = (solver=:mini_batch_sgd, loss=:multinomial_log_loss,
               epochs=model.epochs, updates, objective_history=history,
               l2=T(model.l2), batch_size=model.batch_size, class_order=copy(classes))
    FittedSGDClassifier(model, coefficients, intercept, classes, updates,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context),
        with_class_target(infer_schema(X), classes))
end

partial_fit(model::SGDClassifier, X::AbstractMatrix, y::AbstractVector;
            classes=nothing, kwargs...) =
    _partial_fit_new_sgd_classifier(model, X, y; classes, kwargs...)

function _partial_fit_new_sgd_classifier(model, X, y; classes=nothing, weights=nothing,
                                         context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "SGDClassifier partial fitting")
    _validate_numeric_matrix(X, "SGDClassifier")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "SGDClassifier target must match the observation count."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "SGDClassifier requires observations and features."))
    class_order = classes === nothing ? _classification_classes(y) :
        _classification_classes(collect(classes))
    all(value -> value in class_order, y) || throw(SchemaMismatchError(
        "SGDClassifier target contains a class absent from classes."))
    T = float(eltype(X))
    data = T.(X)
    coefficients = zeros(T, length(class_order), size(X, 2))
    intercept = zeros(T, length(class_order))
    targets = _sgd_class_targets(y, class_order)
    updates, history = _train_sgd_classifier!(model, coefficients, intercept,
        0, data, targets, 1, context)
    details = (solver=:mini_batch_sgd, loss=:multinomial_log_loss,
               epochs=1, updates, objective_history=history, l2=T(model.l2),
               batch_size=model.batch_size, class_order=copy(class_order))
    FittedSGDClassifier(model, coefficients, intercept, class_order, updates,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context),
        with_class_target(infer_schema(X), class_order))
end

function partial_fit(fitted::FittedSGDClassifier, X::AbstractMatrix,
                     y::AbstractVector; weights=nothing, context=default_context())
    reject_unsupported_weights(fitted.model, weights)
    _validate_numeric_matrix(X, "SGDClassifier")
    _validate_feature_count(fitted.schema, X, "SGDClassifier")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "SGDClassifier target must match the observation count."))
    T = eltype(fitted.coefficients)
    coefficients, intercept = copy(fitted.coefficients), copy(fitted.intercept)
    targets = _sgd_class_targets(y, fitted.classes)
    updates, history = _train_sgd_classifier!(fitted.model, coefficients,
        intercept, fitted.updates, T.(X), targets, 1, context)
    details = merge(fitted.report.details,
        (updates=updates, objective_history=vcat(fitted.report.details.objective_history, history),))
    FittedSGDClassifier(fitted.model, coefficients, intercept, fitted.classes, updates,
        FitReport(observations=fitted.report.observations + size(X, 1),
                  features=size(X, 2), details=details, context=context), fitted.schema)
end

function predict(fitted::FittedSGDRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "SGDRegressor")
    _validate_feature_count(fitted.schema, X, "SGDRegressor")
    X * fitted.coefficients .+ fitted.intercept
end

function predict_proba(fitted::FittedSGDClassifier, X::AbstractMatrix)
    _validate_numeric_matrix(X, "SGDClassifier")
    _validate_feature_count(fitted.schema, X, "SGDClassifier")
    Kernels.softmax(X * transpose(fitted.coefficients) .+
                    transpose(fitted.intercept); dims=2)
end

function predict(fitted::FittedSGDClassifier, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))]
     for row in axes(probabilities, 1)]
end

report(fitted::Union{FittedSGDRegressor,FittedSGDClassifier}) = fitted.report
