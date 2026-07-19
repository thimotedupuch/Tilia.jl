# ----------------- MultiOutputRegressor -----------------

struct MultiOutputRegressor{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    function MultiOutputRegressor(estimator::AbstractPredictor)
        capabilities(typeof(estimator)).task === :regression || throw(InvalidHyperparameterError("MultiOutputRegressor requires a regression base estimator."))
        new{typeof(estimator)}(estimator)
    end
end

struct FittedMultiOutputRegressor{M,F,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    report::R
    schema::S
end

capabilities(::Type{<:MultiOutputRegressor}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::MultiOutputRegressor, X::AbstractMatrix, y::AbstractMatrix;
             weights=nothing, context=default_context())
    n, m = size(y)
    estimators = []
    for col in 1:m
        y_col = view(y, :, col)
        fitted = fit(model.estimator, X, y_col; weights=weights, context=context)
        push!(estimators, fitted)
    end
    
    details = (num_outputs=m,)
    fit_report = FitReport(status=:success, observations=size(X,1), features=size(X,2),
                           backend=:cpu, details=details, context=context)
    
    FittedMultiOutputRegressor(model, estimators, fit_report,
                               with_target(infer_schema(X), y[:, 1]))
end

function predict(fitted::FittedMultiOutputRegressor, X::AbstractMatrix)
    n = size(X, 1)
    m = length(fitted.estimators)
    preds = Matrix{Float64}(undef, n, m)
    for col in 1:m
        preds[:, col] = predict(fitted.estimators[col], X)
    end
    return preds
end

report(fitted::FittedMultiOutputRegressor) = fitted.report

