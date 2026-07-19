"""
Generalized Linear Models (GLMs) for Poisson, Gamma, and Tweedie regression.
"""

struct PoissonRegression{T<:Real} <: AbstractPredictor
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    link::Symbol
    function PoissonRegression(; lambda::Real=0.0, fit_intercept::Bool=true,
                               max_iterations::Integer=500, tolerance::Real=1e-6,
                               link::Symbol=:log)
        lambda >= 0 || throw(InvalidHyperparameterError("PoissonRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("PoissonRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("PoissonRegression tolerance must be positive; received $tolerance."))
        link in (:log, :identity) || throw(InvalidHyperparameterError("PoissonRegression link must be :log or :identity; received $link."))
        T = promote_type(typeof(lambda), typeof(tolerance))
        new{T}(T(lambda), fit_intercept, Int(max_iterations), T(tolerance), link)
    end
end

struct GammaRegression{T<:Real} <: AbstractPredictor
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    link::Symbol
    function GammaRegression(; lambda::Real=0.0, fit_intercept::Bool=true,
                             max_iterations::Integer=500, tolerance::Real=1e-6,
                             link::Symbol=:log)
        lambda >= 0 || throw(InvalidHyperparameterError("GammaRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("GammaRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("GammaRegression tolerance must be positive; received $tolerance."))
        link in (:log, :identity) || throw(InvalidHyperparameterError("GammaRegression link must be :log or :identity; received $link."))
        T = promote_type(typeof(lambda), typeof(tolerance))
        new{T}(T(lambda), fit_intercept, Int(max_iterations), T(tolerance), link)
    end
end

struct TweedieRegression{T<:Real} <: AbstractPredictor
    power::T
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    link::Symbol
    function TweedieRegression(; power::Real=1.5, lambda::Real=0.0, fit_intercept::Bool=true,
                               max_iterations::Integer=500, tolerance::Real=1e-6,
                               link::Symbol=:auto)
        power >= 1 || power <= 0 || throw(InvalidHyperparameterError("TweedieRegression power must be >= 1 or <= 0; received $power."))
        lambda >= 0 || throw(InvalidHyperparameterError("TweedieRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("TweedieRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("TweedieRegression tolerance must be positive; received $tolerance."))
        link in (:auto, :log, :identity) || throw(InvalidHyperparameterError("TweedieRegression link must be :auto, :log, or :identity; received $link."))
        T = promote_type(typeof(power), typeof(lambda), typeof(tolerance))
        new{T}(T(power), T(lambda), fit_intercept, Int(max_iterations), T(tolerance), link)
    end
end

struct FittedTweedieRegressor{M,C,I,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    report::R
    schema::S
end

capabilities(::Type{<:PoissonRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)
capabilities(::Type{<:GammaRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)
capabilities(::Type{<:TweedieRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function tweedie_objective(θ, X, y, weights, power, lambda, fit_intercept, link)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[end] : zero(T)

    eta = X * beta .+ b

    if link === :log
        mu = exp.(eta)
    elseif link === :identity
        mu = eta
    else
        throw(ArgumentError("Unknown link function $link"))
    end

    if any(mu_i -> mu_i <= zero(T), mu)
        return T(Inf)
    end

    loss = zero(T)
    if power == 1
        for i in 1:n
            w = weights === nothing ? one(T) : T(weights[i])
            if y[i] > 0
                loss += w * (y[i] * log(y[i] / mu[i]) - (y[i] - mu[i]))
            else
                loss += w * mu[i]
            end
        end
    elseif power == 2
        for i in 1:n
            w = weights === nothing ? one(T) : T(weights[i])
            if y[i] <= 0
                return T(Inf)
            end
            loss += w * (log(mu[i] / y[i]) + y[i] / mu[i] - 1)
        end
    else
        for i in 1:n
            w = weights === nothing ? one(T) : T(weights[i])
            y_i = y[i]
            mu_i = mu[i]
            if y_i < 0
                return T(Inf)
            end
            term1 = y_i > 0 ? (y_i^(2 - power)) / ((1 - power) * (2 - power)) : zero(T)
            term2 = -y_i * (mu_i^(1 - power)) / (1 - power)
            term3 = (mu_i^(2 - power)) / (2 - power)
            loss += w * (term1 + term2 + term3)
        end
    end

    reg = 0.5 * T(lambda) * sum(abs2, beta)
    return loss + reg
end

function tweedie_gradient!(grad, θ, X, y, weights, power, lambda, fit_intercept, link)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[end] : zero(T)

    eta = X * beta .+ b
    if link === :log
        mu = exp.(eta)
    elseif link === :identity
        mu = eta
    else
        throw(ArgumentError("Unknown link function $link"))
    end

    g = similar(eta)
    for i in 1:n
        w = weights === nothing ? one(T) : T(weights[i])
        # Safe clip to avoid division by zero or NaN in powers of extremely small numbers
        mu_safe = max(mu[i], T(1e-12))
        if link === :log
            g[i] = w * (mu_safe^(1 - power)) * (mu[i] - y[i])
        elseif link === :identity
            g[i] = w * (mu_safe^(-power)) * (mu[i] - y[i])
        end
    end

    grad_beta = view(grad, 1:d)
    mul!(grad_beta, transpose(X), g)
    grad_beta .+= T(lambda) .* beta

    if fit_intercept
        grad[end] = sum(g)
    end
    return grad
end

function _fit_tweedie_common(model, X, y, weights, context, power, link)
    model_name = string(nameof(typeof(model)))
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_regression_data(X, y, weights, model_name)

    # Additional target validations
    if power == 2 || (power > 2)
        any(yi -> yi <= 0, y) && throw(UnsupportedDataError("$model_name requires strictly positive target values."))
    else
        any(yi -> yi < 0, y) && throw(UnsupportedDataError("$model_name requires nonnegative target values."))
    end

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)
    y_vec = Vector{T}(y)

    d = size(X_mat, 2)
    initial = zeros(T, model.fit_intercept ? d + 1 : d)

    # Set initial intercept if fit_intercept
    if model.fit_intercept
        # For log link, initialize intercept to log(mean(y)) or similar
        mean_y = weights === nothing ? mean(y_vec) : sum(y_vec .* T.(weights)) / sum(weights)
        if link === :log && mean_y > 0
            initial[end] = log(mean_y)
        elseif link === :identity
            initial[end] = mean_y
        end
    end

    obj = θ -> tweedie_objective(θ, X_mat, y_vec, weights, T(power), T(model.lambda), model.fit_intercept, link)
    grad! = (g, θ) -> tweedie_gradient!(g, θ, X_mat, y_vec, weights, T(power), T(model.lambda), model.fit_intercept, link)

    result = Solvers.lbfgs(obj, grad!, initial;
                           tolerance=effective_tolerance(context, model.tolerance),
                           max_iterations=effective_max_iterations(context, model.max_iterations))

    coefficients = result.parameters[1:d]
    intercept = model.fit_intercept ? result.parameters[end] : zero(T)

    details = (solver=:lbfgs, objective_history=result.objective_history,
               iterations=result.iterations, converged=result.converged,
               power=power, link=link)

    fit_report = FitReport(status=result.converged ? :success : :max_iterations,
                           observations=size(X, 1), features=size(X, 2),
                           backend=:cpu, details=details, context=context)

    FittedTweedieRegressor(model, coefficients, intercept, fit_report,
                           with_target(infer_schema(X), y))
end

function fit(model::PoissonRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    _fit_tweedie_common(model, X, y, weights, context, 1.0, model.link)
end

function fit(model::GammaRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    _fit_tweedie_common(model, X, y, weights, context, 2.0, model.link)
end

function fit(model::TweedieRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    link = model.link === :auto ? (model.power <= 0 ? :identity : :log) : model.link
    _fit_tweedie_common(model, X, y, weights, context, model.power, link)
end

function predict(fitted::FittedTweedieRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    eta = X * fitted.coefficients .+ fitted.intercept

    link = :log
    if hasproperty(fitted.model, :link)
        link_val = fitted.model.link
        if link_val === :auto
            link = (hasproperty(fitted.model, :power) && fitted.model.power <= 0) ? :identity : :log
        else
            link = link_val
        end
    end

    if link === :log
        return exp.(eta)
    elseif link === :identity
        return eta
    else
        throw(ArgumentError("Unknown link function $link"))
    end
end

report(fitted::FittedTweedieRegressor) = fitted.report
