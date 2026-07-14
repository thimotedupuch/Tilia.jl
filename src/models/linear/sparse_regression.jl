abstract type AbstractSparseLinearRegressor <: AbstractPredictor end

"""L1-regularized least-squares regression fitted by coordinate descent."""
struct Lasso{T<:Real} <: AbstractSparseLinearRegressor
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    function Lasso(; lambda::Real=1.0, fit_intercept::Bool=true,
                   max_iterations::Integer=1_000, tolerance::Real=1e-6)
        isfinite(lambda) && lambda >= 0 || throw(InvalidHyperparameterError(
            "Lasso lambda must be finite and nonnegative."))
        max_iterations > 0 || throw(InvalidHyperparameterError("Lasso max_iterations must be positive."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "Lasso tolerance must be finite and positive."))
        T = promote_type(typeof(lambda), typeof(tolerance))
        new{T}(T(lambda), fit_intercept, Int(max_iterations), T(tolerance))
    end
end

"""Elastic-net regression combining L1 and L2 penalties."""
struct ElasticNet{T<:Real} <: AbstractSparseLinearRegressor
    lambda::T
    l1_ratio::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    function ElasticNet(; lambda::Real=1.0, l1_ratio::Real=0.5,
                        fit_intercept::Bool=true, max_iterations::Integer=1_000,
                        tolerance::Real=1e-6)
        isfinite(lambda) && lambda >= 0 || throw(InvalidHyperparameterError(
            "ElasticNet lambda must be finite and nonnegative."))
        isfinite(l1_ratio) && 0 <= l1_ratio <= 1 || throw(InvalidHyperparameterError(
            "ElasticNet l1_ratio must lie in [0, 1]."))
        max_iterations > 0 || throw(InvalidHyperparameterError("ElasticNet max_iterations must be positive."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "ElasticNet tolerance must be finite and positive."))
        T = promote_type(typeof(lambda), typeof(l1_ratio), typeof(tolerance))
        new{T}(T(lambda), T(l1_ratio), fit_intercept, Int(max_iterations), T(tolerance))
    end
end

struct FittedSparseLinearRegressor{M,T,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::Vector{T}
    intercept::T
    report::R
    schema::S
end

capabilities(::Type{<:AbstractSparseLinearRegressor}) = (task=:regression, sparse=true,
    missing=false, weights=true, partial_fit=false, probabilistic=false)

function fit(model::AbstractSparseLinearRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_regression_data(X, y, weights, name)
    T = float(promote_type(eltype(X), eltype(y),
        weights === nothing ? eltype(y) : eltype(weights)))
    data = T.(X)
    l1_penalty, l2_penalty = model isa Lasso ? (T(model.lambda), zero(T)) :
        (T(model.lambda * model.l1_ratio), T(model.lambda * (1 - model.l1_ratio)))
    result = Solvers.elastic_net_coordinate_descent(data, T.(y);
        l1_penalty=l1_penalty, l2_penalty=l2_penalty,
        fit_intercept=model.fit_intercept, weights=weights,
        max_iterations=model.max_iterations, tolerance=T(model.tolerance))
    warnings = result.converged ? String[] : ["$name reached max_iterations without convergence."]
    details = (solver=:coordinate_descent, iterations=result.iterations,
               converged=result.converged, maximum_update=result.maximum_update,
               objective_history=result.objective_history,
               nonzero_coefficients=count(coefficient -> !iszero(coefficient), result.coefficients),
               l1_penalty=l1_penalty, l2_penalty=l2_penalty,
               weighted=weights !== nothing)
    fit_report = FitReport(status=result.converged ? :success : :max_iterations,
        observations=size(X, 1), features=size(X, 2), backend=:cpu,
        warnings=warnings, details=details)
    FittedSparseLinearRegressor(model, result.coefficients, result.intercept,
                                fit_report, infer_schema(X))
end

function predict(fitted::FittedSparseLinearRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedSparseLinearRegressor) = fitted.report
