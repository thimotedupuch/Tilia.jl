"""Kernel ridge regression using a dense dual linear solve."""
struct KernelRidgeRegression{T<:Real} <: AbstractPredictor
    lambda::T
    kernel::Symbol
    gamma::Union{Symbol,T}
    degree::Int
    coef0::T
    function KernelRidgeRegression(; lambda::Real=1.0, kernel::Symbol=:rbf,
            gamma=:scale, degree::Integer=3, coef0::Real=1.0)
        isfinite(lambda) && lambda > 0 || throw(InvalidHyperparameterError(
            "KernelRidgeRegression lambda must be finite and positive."))
        kernel in (:linear, :rbf, :polynomial) || throw(InvalidHyperparameterError(
            "KernelRidgeRegression kernel must be :linear, :rbf, or :polynomial."))
        gamma === :scale || (gamma isa Real && isfinite(gamma) && gamma > 0) ||
            throw(InvalidHyperparameterError("gamma must be :scale or a finite positive number."))
        degree > 0 || throw(InvalidHyperparameterError("degree must be positive."))
        isfinite(coef0) || throw(InvalidHyperparameterError("coef0 must be finite."))
        T = promote_type(typeof(lambda), gamma === :scale ? typeof(lambda) : typeof(gamma), typeof(coef0))
        new{T}(T(lambda), kernel, gamma === :scale ? :scale : T(gamma), Int(degree), T(coef0))
    end
end

struct FittedKernelRidge{M,T,R,S} <: AbstractFittedEstimator
    model::M
    training_data::Matrix{T}
    dual_coefficients::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:KernelRidgeRegression}) = (task=:regression, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function fit(model::KernelRidgeRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    weights === nothing || throw(UnsupportedDataError(
        "KernelRidgeRegression observation weights are not supported."))
    require_cpu(context, "KernelRidgeRegression fitting")
    _validate_regression_data(X, y, nothing, "KernelRidgeRegression")
    T = float(promote_type(eltype(X), eltype(y)))
    data, target = Matrix{T}(X), T.(y)
    kernel = Kernels.gram_matrix(data; kernel=model.kernel, gamma=model.gamma,
                                 degree=model.degree, coef0=model.coef0)
    system = Hermitian(kernel + T(model.lambda) * I)
    coefficients = try
        cholesky(system) \ target
    catch error
        error isa PosDefException || rethrow()
        throw(NumericalFailureError(
            "KernelRidgeRegression kernel system was not positive definite."))
    end
    residual_norm = norm(kernel * coefficients - target)
    details = (kernel=model.kernel, gamma=model.gamma, degree=model.degree,
               regularization=model.lambda, dual_coefficients=length(coefficients),
               residual_norm=residual_norm)
    FittedKernelRidge(model, data, coefficients,
        FitReport(observations=size(X, 1), features=size(X, 2), backend=:cpu,
                  details=details, context=context), with_target(infer_schema(X), y))
end

function predict(fitted::FittedKernelRidge, X::AbstractMatrix)
    _validate_numeric_matrix(X, "KernelRidgeRegression")
    _validate_feature_count(fitted.schema, X, "KernelRidgeRegression")
    kernel = Kernels.gram_matrix(X, fitted.training_data; kernel=fitted.model.kernel,
        gamma=fitted.model.gamma, degree=fitted.model.degree, coef0=fitted.model.coef0)
    kernel * fitted.dual_coefficients
end

report(fitted::FittedKernelRidge) = fitted.report
