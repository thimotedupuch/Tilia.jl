# ----------------- TransformedTargetRegressor -----------------

struct TransformedTargetRegressor{R<:AbstractPredictor, F, FI} <: AbstractPredictor
    regressor::R
    func::F
    inverse_func::FI
    function TransformedTargetRegressor(regressor::AbstractPredictor; func=identity, inverse_func=identity)
        capabilities(typeof(regressor)).task === :regression || throw(InvalidHyperparameterError("TransformedTargetRegressor requires a regression model."))
        new{typeof(regressor), typeof(func), typeof(inverse_func)}(regressor, func, inverse_func)
    end
end

struct FittedTransformedTargetRegressor{M,R,Rp,S} <: AbstractFittedEstimator
    model::M
    fitted_regressor::R
    report::Rp
    schema::S
end

capabilities(::Type{<:TransformedTargetRegressor}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::TransformedTargetRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    y_trans = model.func.(y)
    fitted_regressor = fit(model.regressor, X, y_trans; weights=weights, context=context)
    
    fit_report = FitReport(status=:success, observations=size(X, 1), features=size(X, 2),
                           backend=:cpu, details=(;), context=context)
                           
    FittedTransformedTargetRegressor(model, fitted_regressor, fit_report,
                                     with_target(infer_schema(X), y))
end

function predict(fitted::FittedTransformedTargetRegressor, X::AbstractMatrix)
    preds_trans = predict(fitted.fitted_regressor, X)
    fitted.model.inverse_func.(preds_trans)
end

report(fitted::FittedTransformedTargetRegressor) = fitted.report

