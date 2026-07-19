# ----------------- StackingRegressor -----------------

struct StackingRegressor{E<:Tuple, F<:AbstractPredictor} <: AbstractPredictor
    estimators::E
    final_estimator::F
    cv::KFold
    function StackingRegressor(estimators::Tuple, final_estimator::AbstractPredictor; cv=KFold(5))
        all(est -> capabilities(typeof(est)).task === :regression, estimators) ||
            throw(InvalidHyperparameterError("StackingRegressor estimators must all be regression models."))
        capabilities(typeof(final_estimator)).task === :regression ||
            throw(InvalidHyperparameterError("StackingRegressor final_estimator must be a regression model."))
        new{typeof(estimators), typeof(final_estimator)}(estimators, final_estimator, cv)
    end
end

struct FittedStackingRegressor{M,B,F,R,S} <: AbstractFittedEstimator
    model::M
    estimators::B
    final_estimator::F
    report::R
    schema::S
end

capabilities(::Type{<:StackingRegressor}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::StackingRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    n, d = size(X)
    m = length(model.estimators)
    
    oof = Matrix{Float64}(undef, n, m)
    folds = split(model.cv, n)
    
    for (idx, est) in enumerate(model.estimators)
        for (train_idx, test_idx) in folds
            X_tr = X[train_idx, :]
            y_tr = y[train_idx]
            w_tr = weights === nothing ? nothing : weights[train_idx]
            
            fitted_fold = fit(est, X_tr, y_tr; weights=w_tr, context=context)
            oof[test_idx, idx] = predict(fitted_fold, X[test_idx, :])
        end
    end
    
    fitted_final = fit(model.final_estimator, oof, y; weights=weights, context=context)
    fitted_base = [fit(est, X, y; weights=weights, context=context) for est in model.estimators]
    
    details = (num_estimators=m, solver=:stacking)
    fit_report = FitReport(status=:success, observations=n, features=d,
                           backend=:cpu, details=details, context=context)
                           
    FittedStackingRegressor(model, fitted_base, fitted_final, fit_report,
                            with_target(infer_schema(X), y))
end

function predict(fitted::FittedStackingRegressor, X::AbstractMatrix)
    n = size(X, 1)
    m = length(fitted.estimators)
    base_preds = Matrix{Float64}(undef, n, m)
    for idx in 1:m
        base_preds[:, idx] = predict(fitted.estimators[idx], X)
    end
    predict(fitted.final_estimator, base_preds)
end

report(fitted::FittedStackingRegressor) = fitted.report

