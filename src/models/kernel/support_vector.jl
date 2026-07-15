abstract type AbstractSupportVectorModel <: AbstractPredictor end

"""Kernel classifier minimizing a regularized squared-hinge objective."""
struct SupportVectorClassifier{T<:Real} <: AbstractSupportVectorModel
    C::T
    kernel::Symbol
    gamma::Union{Symbol,T}
    degree::Int
    coef0::T
    max_iterations::Int
    tolerance::T
    function SupportVectorClassifier(; C::Real=1.0, kernel::Symbol=:rbf,
            gamma=:scale, degree::Integer=3, coef0::Real=1.0,
            max_iterations::Integer=2_000, tolerance::Real=1e-6)
        parameters = _validate_support_vector_parameters(C, kernel, gamma, degree,
            coef0, max_iterations, tolerance, "SupportVectorClassifier")
        new{parameters.T}(parameters.C, kernel, parameters.gamma, Int(degree),
            parameters.coef0, Int(max_iterations), parameters.tolerance)
    end
end

"""Kernel regressor minimizing a regularized squared epsilon-insensitive objective."""
struct SupportVectorRegressor{T<:Real} <: AbstractSupportVectorModel
    C::T
    epsilon::T
    kernel::Symbol
    gamma::Union{Symbol,T}
    degree::Int
    coef0::T
    max_iterations::Int
    tolerance::T
    function SupportVectorRegressor(; C::Real=1.0, epsilon::Real=0.1,
            kernel::Symbol=:rbf, gamma=:scale, degree::Integer=3,
            coef0::Real=1.0, max_iterations::Integer=2_000,
            tolerance::Real=1e-6)
        isfinite(epsilon) && epsilon >= 0 || throw(InvalidHyperparameterError(
            "SupportVectorRegressor epsilon must be finite and nonnegative."))
        parameters = _validate_support_vector_parameters(C, kernel, gamma, degree,
            coef0, max_iterations, tolerance, "SupportVectorRegressor")
        new{parameters.T}(parameters.C, parameters.T(epsilon), kernel,
            parameters.gamma, Int(degree), parameters.coef0,
            Int(max_iterations), parameters.tolerance)
    end
end

function _validate_support_vector_parameters(C, kernel, gamma, degree, coef0,
                                             max_iterations, tolerance, name)
    isfinite(C) && C > 0 || throw(InvalidHyperparameterError("$name C must be finite and positive."))
    kernel in (:linear, :rbf, :polynomial) || throw(InvalidHyperparameterError(
        "$name kernel must be :linear, :rbf, or :polynomial."))
    gamma === :scale || (gamma isa Real && isfinite(gamma) && gamma > 0) ||
        throw(InvalidHyperparameterError("$name gamma must be :scale or a finite positive number."))
    degree > 0 || throw(InvalidHyperparameterError("$name degree must be positive."))
    isfinite(coef0) || throw(InvalidHyperparameterError("$name coef0 must be finite."))
    max_iterations > 0 || throw(InvalidHyperparameterError("$name max_iterations must be positive."))
    isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
        "$name tolerance must be finite and positive."))
    T = promote_type(typeof(C), gamma === :scale ? typeof(C) : typeof(gamma),
                     typeof(coef0), typeof(tolerance))
    (T=T, C=T(C), gamma=gamma === :scale ? :scale : T(gamma),
     coef0=T(coef0), tolerance=T(tolerance))
end

struct FittedSupportVectorClassifier{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    training_data::Matrix{T}
    coefficients::Matrix{T}
    intercept::Vector{T}
    classes::Vector{L}
    report::R
    schema::S
end

struct FittedSupportVectorRegressor{M,T,R,S} <: AbstractFittedEstimator
    model::M
    training_data::Matrix{T}
    coefficients::Vector{T}
    intercept::T
    report::R
    schema::S
end

capabilities(::Type{<:SupportVectorClassifier}) = (task=:classification, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=false)
capabilities(::Type{<:SupportVectorRegressor}) = (task=:regression, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=false)

function _kernel_optimizer_step(kernel, weights, C, T)
    augmented = hcat(kernel, ones(T, size(kernel, 1)))
    spectral_kernel = opnorm(Symmetric(kernel))
    loss_lipschitz = C / sum(weights) * maximum(weights) * opnorm(augmented)^2
    inv(max(spectral_kernel + loss_lipschitz, eps(T)))
end

function _fit_squared_hinge(kernel, target, weights, C, max_iterations, tolerance, T)
    coefficients = zeros(T, size(kernel, 1))
    intercept = zero(T)
    step = _kernel_optimizer_step(kernel, weights, C, T)
    history = T[]
    converged = false
    iterations = max_iterations
    maximum_update = T(Inf)
    for iteration in 1:max_iterations
        scores = kernel * coefficients .+ intercept
        violations = max.(one(T) .- target .* scores, zero(T))
        derivative = -(T(C) / sum(weights)) .* weights .* target .* violations
        coefficient_gradient = kernel * coefficients .+ transpose(kernel) * derivative
        intercept_gradient = sum(derivative)
        updated = coefficients .- step .* coefficient_gradient
        new_intercept = intercept - step * intercept_gradient
        maximum_update = max(maximum(abs.(updated .- coefficients); init=zero(T)),
                             abs(new_intercept - intercept))
        coefficients, intercept = updated, new_intercept
        objective = dot(coefficients, kernel * coefficients) / T(2) +
                    T(C) * sum(weights .* abs2.(max.(one(T) .-
                    target .* (kernel * coefficients .+ intercept), zero(T)))) /
                    (T(2) * sum(weights))
        push!(history, objective)
        if maximum_update <= T(tolerance)
            converged = true
            iterations = iteration
            break
        end
    end
    (coefficients=coefficients, intercept=intercept, history=history,
     converged=converged, iterations=iterations, maximum_update=maximum_update)
end

function fit(model::SupportVectorClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "SupportVectorClassifier fitting")
    _validate_numeric_matrix(X, "SupportVectorClassifier")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "SupportVectorClassifier target has length $(length(y)); expected $(size(X, 1))."))
    classes = _classification_classes(y)
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = Matrix{T}(X)
    observation_weights = _boosting_weights(weights, size(X, 1), T, "SupportVectorClassifier")
    kernel = Kernels.gram_matrix(data; kernel=model.kernel, gamma=model.gamma,
                                 degree=model.degree, coef0=model.coef0)
    trained_classes = length(classes) == 2 ? classes[end:end] : classes
    results = [_fit_squared_hinge(kernel, ifelse.(y .== class, one(T), -one(T)),
        observation_weights, T(model.C), model.max_iterations, T(model.tolerance), T)
        for class in trained_classes]
    coefficients = reduce(hcat, [result.coefficients for result in results])
    intercept = T[result.intercept for result in results]
    convergence = [result.converged for result in results]
    details = (solver=:gradient_descent, loss=:squared_hinge,
        convergence=convergence, iterations=[result.iterations for result in results],
        objective_history=[result.history for result in results],
        support_vectors=[count(value -> abs(value) > T(model.tolerance), result.coefficients)
                         for result in results], kernel=model.kernel,
        class_order=copy(classes), strategy=:one_vs_rest)
    schema = infer_schema(X)
    schema = Schema(schema.columns; class_order=Any[classes...])
    FittedSupportVectorClassifier(model, data, coefficients, intercept, classes,
        FitReport(status=all(convergence) ? :success : :max_iterations,
            observations=size(X, 1), features=size(X, 2), backend=:cpu,
            warnings=all(convergence) ? String[] : ["One or more support-vector objectives did not converge."],
            details=details, context=context), schema)
end

function _fit_epsilon_insensitive(kernel, target, weights, model, T)
    coefficients = zeros(T, size(kernel, 1))
    intercept = sum(weights .* target) / sum(weights)
    step = _kernel_optimizer_step(kernel, weights, T(model.C), T)
    history = T[]
    converged = false
    iterations = model.max_iterations
    maximum_update = T(Inf)
    for iteration in 1:model.max_iterations
        residual = kernel * coefficients .+ intercept .- target
        excess = max.(abs.(residual) .- T(model.epsilon), zero(T))
        derivative = T(model.C) / sum(weights) .* weights .* sign.(residual) .* excess
        coefficient_gradient = kernel * coefficients .+ transpose(kernel) * derivative
        intercept_gradient = sum(derivative)
        updated = coefficients .- step .* coefficient_gradient
        new_intercept = intercept - step * intercept_gradient
        maximum_update = max(maximum(abs.(updated .- coefficients); init=zero(T)),
                             abs(new_intercept - intercept))
        coefficients, intercept = updated, new_intercept
        residual = kernel * coefficients .+ intercept .- target
        objective = dot(coefficients, kernel * coefficients) / T(2) +
                    T(model.C) * sum(weights .* abs2.(max.(abs.(residual) .-
                    T(model.epsilon), zero(T)))) / (T(2) * sum(weights))
        push!(history, objective)
        if maximum_update <= T(model.tolerance)
            converged = true
            iterations = iteration
            break
        end
    end
    (coefficients=coefficients, intercept=intercept, history=history,
     converged=converged, iterations=iterations, maximum_update=maximum_update)
end

function fit(model::SupportVectorRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "SupportVectorRegressor fitting")
    _validate_regression_data(X, y, weights, "SupportVectorRegressor")
    T = float(promote_type(eltype(X), eltype(y), weights === nothing ? eltype(y) : eltype(weights)))
    data, target = Matrix{T}(X), T.(y)
    observation_weights = _boosting_weights(weights, size(X, 1), T, "SupportVectorRegressor")
    kernel = Kernels.gram_matrix(data; kernel=model.kernel, gamma=model.gamma,
                                 degree=model.degree, coef0=model.coef0)
    result = _fit_epsilon_insensitive(kernel, target, observation_weights, model, T)
    details = (solver=:gradient_descent, loss=:squared_epsilon_insensitive,
        converged=result.converged, iterations=result.iterations,
        objective_history=result.history,
        support_vectors=count(value -> abs(value) > T(model.tolerance), result.coefficients),
        kernel=model.kernel, epsilon=model.epsilon)
    FittedSupportVectorRegressor(model, data, result.coefficients, result.intercept,
        FitReport(status=result.converged ? :success : :max_iterations,
            observations=size(X, 1), features=size(X, 2), backend=:cpu,
            warnings=result.converged ? String[] : ["Support-vector objective did not converge."],
            details=details, context=context), infer_schema(X))
end

function _support_scores(fitted, X)
    Kernels.gram_matrix(X, fitted.training_data; kernel=fitted.model.kernel,
        gamma=fitted.model.gamma, degree=fitted.model.degree, coef0=fitted.model.coef0) *
        fitted.coefficients .+ transpose(fitted.intercept)
end

function predict(fitted::FittedSupportVectorClassifier, X::AbstractMatrix)
    _validate_numeric_matrix(X, "SupportVectorClassifier")
    _validate_feature_count(fitted.schema, X, "SupportVectorClassifier")
    scores = _support_scores(fitted, X)
    if length(fitted.classes) == 2
        return ifelse.(vec(scores) .>= 0, fitted.classes[2], fitted.classes[1])
    end
    [fitted.classes[argmax(view(scores, row, :))] for row in axes(X, 1)]
end

function predict(fitted::FittedSupportVectorRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "SupportVectorRegressor")
    _validate_feature_count(fitted.schema, X, "SupportVectorRegressor")
    vec(_support_scores(fitted, X))
end

report(fitted::Union{FittedSupportVectorClassifier,FittedSupportVectorRegressor}) = fitted.report
