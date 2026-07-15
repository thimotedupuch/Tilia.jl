"""Elastic-net logistic regression fitted by proximal gradient descent."""
struct SparseLogisticRegression{T<:Real} <: AbstractPredictor
    lambda::T
    l1_ratio::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    function SparseLogisticRegression(; lambda::Real=1.0, l1_ratio::Real=1.0,
                                      fit_intercept::Bool=true,
                                      max_iterations::Integer=1_000,
                                      tolerance::Real=1e-6)
        isfinite(lambda) && lambda >= 0 || throw(InvalidHyperparameterError(
            "SparseLogisticRegression lambda must be finite and nonnegative."))
        isfinite(l1_ratio) && 0 <= l1_ratio <= 1 || throw(InvalidHyperparameterError(
            "SparseLogisticRegression l1_ratio must lie in [0, 1]."))
        max_iterations > 0 || throw(InvalidHyperparameterError(
            "SparseLogisticRegression max_iterations must be positive."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "SparseLogisticRegression tolerance must be finite and positive."))
        T = promote_type(typeof(lambda), typeof(l1_ratio), typeof(tolerance))
        new{T}(T(lambda), T(l1_ratio), fit_intercept, Int(max_iterations), T(tolerance))
    end
end

struct FittedSparseLogisticRegression{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::Matrix{T}
    intercept::Vector{T}
    classes::Vector{L}
    report::R
    schema::S
end

capabilities(::Type{<:SparseLogisticRegression}) = (task=:classification, sparse=true,
    missing=false, weights=true, partial_fit=false, probabilistic=true)

function _sparse_logistic_objective(X, target, weights, coefficients, intercept,
                                    l1_penalty, l2_penalty)
    logits = X * coefficients .+ intercept
    losses = max.(logits, zero(eltype(logits))) .- logits .* target .+
             log1p.(exp.(-abs.(logits)))
    sum(weights .* losses) / sum(weights) + l1_penalty * sum(abs, coefficients) +
        l2_penalty * sum(abs2, coefficients) / 2
end

function _fit_sparse_logistic_binary(model, X, target, weights, T)
    coefficients = zeros(T, size(X, 2))
    positive_rate = clamp(sum(weights .* target) / sum(weights), eps(T), one(T) - eps(T))
    intercept = model.fit_intercept ? log(positive_rate / (one(T) - positive_rate)) : zero(T)
    l1_penalty = T(model.lambda * model.l1_ratio)
    l2_penalty = T(model.lambda * (1 - model.l1_ratio))
    row_norms = vec(sum(abs2, X; dims=2))
    lipschitz = sum(weights .* row_norms) / (T(4) * sum(weights)) + l2_penalty
    model.fit_intercept && (lipschitz += T(0.25))
    step = inv(max(lipschitz, eps(T)))
    history = T[]
    converged = false
    iterations = model.max_iterations
    maximum_update = T(Inf)
    for iteration in 1:model.max_iterations
        probabilities = Kernels.sigmoid(X * coefficients .+ intercept)
        residual = weights .* (probabilities .- target) ./ sum(weights)
        gradient = transpose(X) * residual .+ l2_penalty .* coefficients
        candidate = coefficients .- step .* gradient
        updated = sign.(candidate) .* max.(abs.(candidate) .- step * l1_penalty, zero(T))
        intercept_update = model.fit_intercept ? step * sum(residual) : zero(T)
        new_intercept = intercept - intercept_update
        maximum_update = max(maximum(abs.(updated .- coefficients); init=zero(T)),
                             abs(new_intercept - intercept))
        coefficients = updated
        intercept = new_intercept
        push!(history, _sparse_logistic_objective(X, target, weights, coefficients,
                                                  intercept, l1_penalty, l2_penalty))
        if maximum_update <= T(model.tolerance)
            converged = true
            iterations = iteration
            break
        end
    end
    (coefficients=coefficients, intercept=intercept, history=history,
     converged=converged, iterations=iterations, maximum_update=maximum_update,
     l1_penalty=l1_penalty, l2_penalty=l2_penalty)
end

function fit(model::SparseLogisticRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "SparseLogisticRegression fitting")
    _validate_numeric_matrix(X, "SparseLogisticRegression")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "SparseLogisticRegression target has length $(length(y)); expected $(size(X, 1))."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "SparseLogisticRegression requires at least one observation and feature."))
    classes = _classification_classes(y)
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = T.(X)
    observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
    length(observation_weights) == length(y) || throw(SchemaMismatchError(
        "SparseLogisticRegression weights have length $(length(observation_weights)); expected $(length(y))."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) &&
        sum(observation_weights) > 0 || throw(UnsupportedDataError(
        "SparseLogisticRegression weights must be finite, nonnegative, and have positive sum."))
    trained_classes = length(classes) == 2 ? classes[end:end] : classes
    coefficients = Matrix{T}(undef, size(X, 2), length(trained_classes))
    intercepts = Vector{T}(undef, length(trained_classes))
    results = map(trained_classes) do class
        _fit_sparse_logistic_binary(model, data, T.(y .== class), observation_weights, T)
    end
    for index in eachindex(results)
        coefficients[:, index] .= results[index].coefficients
        intercepts[index] = results[index].intercept
    end
    converged = [result.converged for result in results]
    warnings = all(converged) ? String[] :
        ["One or more sparse logistic objectives reached max_iterations."]
    details = (solver=:proximal_gradient, convergence=converged,
        iterations=[result.iterations for result in results],
        objective_history=[result.history for result in results],
        maximum_updates=[result.maximum_update for result in results],
        nonzero_coefficients=count(coefficient -> !iszero(coefficient), coefficients),
        l1_penalty=first(results).l1_penalty, l2_penalty=first(results).l2_penalty,
        class_order=copy(classes), strategy=:one_vs_rest)
    schema = infer_schema(X)
    schema = Schema(schema.columns; class_order=Any[classes...])
    FittedSparseLogisticRegression(model, coefficients, intercepts, classes,
        FitReport(status=all(converged) ? :success : :max_iterations,
            observations=size(X, 1), features=size(X, 2), backend=:cpu,
            warnings=warnings, details=details, context=context), schema)
end

function predict_proba(fitted::FittedSparseLogisticRegression, X::AbstractMatrix)
    _validate_numeric_matrix(X, "SparseLogisticRegression")
    _validate_feature_count(fitted.schema, X, "SparseLogisticRegression")
    positive = Kernels.sigmoid(X * fitted.coefficients .+ transpose(fitted.intercept))
    length(fitted.classes) == 2 && return hcat(one(eltype(positive)) .- vec(positive), vec(positive))
    positive ./ sum(positive; dims=2)
end

function predict(fitted::FittedSparseLogisticRegression, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

report(fitted::FittedSparseLogisticRegression) = fitted.report
