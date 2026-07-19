"""
Ordinal regression (proportional odds model) using a cumulative link formulation.
"""

struct OrdinalRegression{T<:Real} <: AbstractPredictor
    lambda::T
    max_iterations::Int
    tolerance::T
    function OrdinalRegression(; lambda::Real=0.0, max_iterations::Integer=500,
                               tolerance::Real=1e-6)
        lambda >= 0 || throw(InvalidHyperparameterError("OrdinalRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("OrdinalRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("OrdinalRegression tolerance must be positive; received $tolerance."))
        T = promote_type(typeof(lambda), typeof(tolerance))
        new{T}(T(lambda), Int(max_iterations), T(tolerance))
    end
end

struct FittedOrdinalRegressor{M,C,A,L,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    alpha::A
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:OrdinalRegression}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function ordinal_objective(θ_param, X, y_int, weights, lambda, K)
    T = eltype(θ_param)
    N, D = size(X)
    beta = view(θ_param, 1:D)
    alpha = view(θ_param, D+1:D+K-1)

    theta = Vector{T}(undef, K-1)
    if K > 1
        theta[1] = alpha[1]
        for j in 2:K-1
            theta[j] = theta[j-1] + exp(alpha[j])
        end
    end

    eta = X * beta

    loss = zero(T)
    for i in 1:N
        w = weights === nothing ? one(T) : T(weights[i])
        ki = y_int[i]

        s_upper = ki == K ? one(T) : Kernels.sigmoid(theta[ki] - eta[i])
        s_lower = ki == 1 ? zero(T) : Kernels.sigmoid(theta[ki-1] - eta[i])

        p_i = max(s_upper - s_lower, T(1e-15))
        loss -= w * log(p_i)
    end

    reg = 0.5 * T(lambda) * sum(abs2, beta)
    return loss + reg
end

function ordinal_gradient!(grad, θ_param, X, y_int, weights, lambda, K)
    T = eltype(θ_param)
    N, D = size(X)
    beta = view(θ_param, 1:D)
    alpha = view(θ_param, D+1:D+K-1)

    theta = Vector{T}(undef, K-1)
    if K > 1
        theta[1] = alpha[1]
        for j in 2:K-1
            theta[j] = theta[j-1] + exp(alpha[j])
        end
    end

    eta = X * beta

    d_eta = zeros(T, N)
    d_theta = zeros(T, K-1)

    for i in 1:N
        w = weights === nothing ? one(T) : T(weights[i])
        ki = y_int[i]

        s_upper = ki == K ? one(T) : Kernels.sigmoid(theta[ki] - eta[i])
        s_lower = ki == 1 ? zero(T) : Kernels.sigmoid(theta[ki-1] - eta[i])

        p_i = max(s_upper - s_lower, T(1e-15))

        ds_upper = ki == K ? zero(T) : s_upper * (one(T) - s_upper)
        ds_lower = ki == 1 ? zero(T) : s_lower * (one(T) - s_lower)

        dp_deta = -ds_upper + ds_lower
        d_eta[i] = -w * dp_deta / p_i

        if ki < K
            d_theta[ki] -= w * ds_upper / p_i
        end
        if ki > 1
            d_theta[ki-1] += w * ds_lower / p_i
        end
    end

    grad_beta = view(grad, 1:D)
    mul!(grad_beta, transpose(X), d_eta)
    grad_beta .+= T(lambda) .* beta

    grad_alpha = view(grad, D+1:D+K-1)
    fill!(grad_alpha, zero(T))
    if K > 1
        grad_alpha[1] = sum(view(d_theta, 1:K-1))

        for m in 2:K-1
            factor = exp(alpha[m])
            grad_alpha[m] = sum(view(d_theta, m:K-1)) * factor
        end
    end

    return grad
end

function fit(model::OrdinalRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    model_name = "OrdinalRegression"
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_numeric_matrix(X, model_name)
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "$model_name target has length $(length(y)); expected $(size(X, 1)) observations."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "$model_name requires at least one observation and feature."))

    classes = _classification_classes(y)
    K = length(classes)

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)

    class_map = Dict(c => idx for (idx, c) in enumerate(classes))
    y_int = [class_map[val] for val in y]

    d = size(X_mat, 2)
    initial = zeros(T, d + K - 1)

    if K > 1
        initial[d+1] = -T(0.5) * (K - 1)
    end

    obj = θ -> ordinal_objective(θ, X_mat, y_int, weights, T(model.lambda), K)
    grad! = (g, θ) -> ordinal_gradient!(g, θ, X_mat, y_int, weights, T(model.lambda), K)

    result = Solvers.lbfgs(obj, grad!, initial;
                           tolerance=effective_tolerance(context, model.tolerance),
                           max_iterations=effective_max_iterations(context, model.max_iterations))

    coefficients = result.parameters[1:d]
    alpha = result.parameters[d+1:end]

    details = (solver=:lbfgs, objective_history=result.objective_history,
               iterations=result.iterations, converged=result.converged,
               classes=copy(classes))

    fit_report = FitReport(status=result.converged ? :success : :max_iterations,
                           observations=size(X, 1), features=size(X, 2),
                           backend=:cpu, details=details, context=context)

    schema = with_class_target(infer_schema(X), classes)
    FittedOrdinalRegressor(model, coefficients, alpha, classes, fit_report, schema)
end

function predict_proba(fitted::FittedOrdinalRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "OrdinalRegression")
    _validate_feature_count(fitted.schema, X, "OrdinalRegression")
    T = eltype(X)
    n = size(X, 1)
    K = length(fitted.classes)

    theta = Vector{T}(undef, K-1)
    if K > 1
        theta[1] = fitted.alpha[1]
        for j in 2:K-1
            theta[j] = theta[j-1] + exp(fitted.alpha[j])
        end
    end

    eta = X * fitted.coefficients
    probs = Matrix{T}(undef, n, K)
    for i in 1:n
        for k in 1:K
            s_upper = k == K ? one(T) : Kernels.sigmoid(theta[k] - eta[i])
            s_lower = k == 1 ? zero(T) : Kernels.sigmoid(theta[k-1] - eta[i])
            probs[i, k] = max(s_upper - s_lower, zero(T))
        end
        probs[i, :] ./= sum(probs[i, :])
    end
    return probs
end

function predict(fitted::FittedOrdinalRegressor, X::AbstractMatrix)
    probs = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
end

report(fitted::FittedOrdinalRegressor) = fitted.report
