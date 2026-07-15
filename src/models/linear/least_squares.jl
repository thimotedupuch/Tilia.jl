"""
Ordinary least-squares regression.

`fit_intercept=true` estimates an unpenalized intercept after centering. Choose
`:qr` for the default pivoted-QR solution or `:svd` for the minimum-norm
rank-deficient solution.
"""
struct LinearRegression <: AbstractPredictor
    fit_intercept::Bool
    solver::Symbol
    function LinearRegression(; fit_intercept::Bool=true, solver::Symbol=:qr)
        solver in (:qr, :svd) || throw(InvalidHyperparameterError(
            "LinearRegression solver must be :qr or :svd; received $solver."))
        new(fit_intercept, solver)
    end
end

"""Ridge regression minimizing `||y-Xβ-b||² + λ||β||²`."""
struct RidgeRegression{T<:Real} <: AbstractPredictor
    lambda::T
    fit_intercept::Bool
    solver::Symbol
    function RidgeRegression(; lambda::Real=1.0, fit_intercept::Bool=true,
                             solver::Symbol=:cholesky)
        isfinite(lambda) && lambda >= 0 || throw(InvalidHyperparameterError(
            "RidgeRegression lambda must be finite and nonnegative; received $lambda."))
        solver in (:cholesky, :svd) || throw(InvalidHyperparameterError(
            "RidgeRegression solver must be :cholesky or :svd; received $solver."))
        new{typeof(lambda)}(lambda, fit_intercept, solver)
    end
end

struct FittedLinearRegressor{M,C,I,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    report::R
    schema::S
end

capabilities(::Type{<:LinearRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)
capabilities(::Type{<:RidgeRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function _validate_regression_data(X, y, weights, model_name)
    _validate_numeric_matrix(X, model_name)
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "$model_name target has length $(length(y)); expected $(size(X, 1)) observations."))
    size(X, 1) > 0 || throw(UnsupportedDataError("$model_name requires at least one observation."))
    size(X, 2) > 0 || throw(UnsupportedDataError("$model_name requires at least one feature."))
    eltype(y) <: Number && all(isfinite, y) || throw(UnsupportedDataError(
        "$model_name requires a finite numeric target."))
    if weights !== nothing
        length(weights) == length(y) || throw(SchemaMismatchError(
            "$model_name weights have length $(length(weights)); expected $(length(y))."))
        all(weight -> isfinite(weight) && weight >= 0, weights) || throw(UnsupportedDataError(
            "$model_name weights must be finite and nonnegative."))
        sum(weights) > 0 || throw(UnsupportedDataError("$model_name weights must have a positive sum."))
    end
end

function _center_regression(X, y, weights, fit_intercept)
    T = float(promote_type(eltype(X), eltype(y), weights === nothing ? eltype(y) : eltype(weights)))
    design = Matrix{T}(X)
    target = Vector{T}(y)
    if fit_intercept
        if weights === nothing
            feature_means = vec(mean(design; dims=1))
            target_mean = mean(target)
        else
            total = sum(weights)
            feature_means = vec(sum(design .* weights; dims=1)) / total
            target_mean = sum(target .* weights) / total
        end
        design .-= transpose(feature_means)
        target .-= target_mean
    else
        feature_means = zeros(T, size(design, 2))
        target_mean = zero(T)
    end
    if weights !== nothing
        square_roots = sqrt.(T.(weights))
        design .*= square_roots
        target .*= square_roots
    end
    design, target, feature_means, target_mean
end

function _fit_linear(model, X, y, weights, context)
    model_name = string(nameof(typeof(model)))
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError(
        "$model_name sparse fitting is not supported yet; provide a dense matrix."))
    _validate_regression_data(X, y, weights, model_name)
    design, target, feature_means, target_mean =
        _center_regression(X, y, weights, model.fit_intercept)
    result = if model isa LinearRegression
        Solvers.least_squares(design, target; solver=model.solver,
                              tolerance=context.numerics.tolerance)
    else
        Solvers.ridge_least_squares(design, target, eltype(design)(model.lambda); solver=model.solver,
                                    tolerance=context.numerics.tolerance)
    end
    intercept = target_mean - dot(feature_means, result.coefficients)
    details = (solver=result.solver, numerical_rank=result.rank,
               residual_norm=result.residual_norm,
               regularization=model isa RidgeRegression ? model.lambda : zero(eltype(result.coefficients)),
               weighted=weights !== nothing, fit_intercept=model.fit_intercept)
    fit_report = FitReport(observations=size(X, 1), features=size(X, 2),
                           backend=:cpu, details=details, context=context)
    FittedLinearRegressor(model, result.coefficients, intercept, fit_report, infer_schema(X))
end

fit(model::Union{LinearRegression,RidgeRegression}, X::AbstractMatrix, y::AbstractVector;
    weights=nothing, context=default_context()) = _fit_linear(model, X, y, weights, context)

function predict(fitted::FittedLinearRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedLinearRegressor) = fitted.report
