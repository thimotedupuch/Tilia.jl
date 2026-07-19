# ----------------- StackingClassifier -----------------

struct StackingClassifier{E<:Tuple, F<:AbstractPredictor} <: AbstractPredictor
    estimators::E
    final_estimator::F
    cv::KFold
    function StackingClassifier(estimators::Tuple, final_estimator::AbstractPredictor; cv=KFold(5))
        all(est -> capabilities(typeof(est)).task === :classification, estimators) ||
            throw(InvalidHyperparameterError("StackingClassifier estimators must all be classification models."))
        capabilities(typeof(final_estimator)).task === :classification ||
            throw(InvalidHyperparameterError("StackingClassifier final_estimator must be a classification model."))
        new{typeof(estimators), typeof(final_estimator)}(estimators, final_estimator, cv)
    end
end

struct FittedStackingClassifier{M,B,F,L,R,S} <: AbstractFittedEstimator
    model::M
    estimators::B
    final_estimator::F
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:StackingClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function _oof_features_for_estimator(est, X_test, fitted_fold)
    if capabilities(typeof(est)).probabilistic
        return predict_proba(fitted_fold, X_test)
    else
        preds = predict(fitted_fold, X_test)
        classes = fitted_fold.classes
        mapping = Dict(c => Float64(idx) for (idx, c) in enumerate(classes))
        return reshape([mapping[val] for val in preds], :, 1)
    end
end

function fit(model::StackingClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    n, d = size(X)
    m = length(model.estimators)
    
    folds = split(model.cv, n)
    
    oof_dims = Int[]
    for (idx, est) in enumerate(model.estimators)
        tr_idx, ts_idx = folds[1]
        fitted_fold = fit(est, X[tr_idx, :], y[tr_idx]; weights=(weights === nothing ? nothing : weights[tr_idx]), context=context)
        feats = _oof_features_for_estimator(est, X[ts_idx, :], fitted_fold)
        push!(oof_dims, size(feats, 2))
    end
    
    total_oof_cols = sum(oof_dims)
    oof = Matrix{Float64}(undef, n, total_oof_cols)
    
    for (idx, est) in enumerate(model.estimators)
        col_start = sum(oof_dims[1:idx-1]) + 1
        col_end = col_start + oof_dims[idx] - 1
        
        for (train_idx, test_idx) in folds
            X_tr = X[train_idx, :]
            y_tr = y[train_idx]
            w_tr = weights === nothing ? nothing : weights[train_idx]
            
            fitted_fold = fit(est, X_tr, y_tr; weights=w_tr, context=context)
            feats = _oof_features_for_estimator(est, X[test_idx, :], fitted_fold)
            oof[test_idx, col_start:col_end] = feats
        end
    end
    
    fitted_final = fit(model.final_estimator, oof, y; weights=weights, context=context)
    fitted_base = [fit(est, X, y; weights=weights, context=context) for est in model.estimators]
    
    details = (num_estimators=m, oof_dims=oof_dims, classes=copy(classes))
    fit_report = FitReport(status=:success, observations=n, features=d,
                           backend=:cpu, details=details, context=context)
                           
    FittedStackingClassifier(model, fitted_base, fitted_final, classes, fit_report,
                             with_class_target(infer_schema(X), classes))
end

function _predict_features_for_stacking(fitted, X)
    n = size(X, 1)
    oof_dims = fitted.report.details.oof_dims
    total_oof_cols = sum(oof_dims)
    base_preds = Matrix{Float64}(undef, n, total_oof_cols)
    
    for idx in 1:length(fitted.estimators)
        col_start = sum(oof_dims[1:idx-1]) + 1
        col_end = col_start + oof_dims[idx] - 1
        
        est = fitted.model.estimators[idx]
        fitted_est = fitted.estimators[idx]
        if capabilities(typeof(est)).probabilistic
            base_preds[:, col_start:col_end] = predict_proba(fitted_est, X)
        else
            preds = predict(fitted_est, X)
            classes = fitted_est.classes
            mapping = Dict(c => Float64(k) for (k, c) in enumerate(classes))
            base_preds[:, col_start:col_end] = reshape([mapping[val] for val in preds], :, 1)
        end
    end
    return base_preds
end

function predict_proba(fitted::FittedStackingClassifier, X::AbstractMatrix)
    base_preds = _predict_features_for_stacking(fitted, X)
    predict_proba(fitted.final_estimator, base_preds)
end

function predict(fitted::FittedStackingClassifier, X::AbstractMatrix)
    base_preds = _predict_features_for_stacking(fitted, X)
    predict(fitted.final_estimator, base_preds)
end

report(fitted::FittedStackingClassifier) = fitted.report

