# ----------------- ThresholdSelectionWrapper -----------------

struct ThresholdSelectionWrapper{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    metric::Symbol
    cv::KFold
    function ThresholdSelectionWrapper(estimator::AbstractPredictor; metric::Symbol=:f1, cv=KFold(5))
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("ThresholdSelectionWrapper requires a classification model."))
        metric in (:f1, :accuracy, :balanced_accuracy) || throw(InvalidHyperparameterError("ThresholdSelectionWrapper metric must be :f1, :accuracy, or :balanced_accuracy."))
        new{typeof(estimator)}(estimator, metric, cv)
    end
end

struct FittedThresholdSelectionWrapper{M,F,T,L,R,S} <: AbstractFittedEstimator
    model::M
    fitted_estimator::F
    threshold::T
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:ThresholdSelectionWrapper}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::ThresholdSelectionWrapper, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    length(classes) == 2 || throw(UnsupportedDataError("ThresholdSelectionWrapper is currently supported only for binary classification."))
    n = size(X, 1)
    
    folds = split(model.cv, n)
    oof_p = Vector{Float64}(undef, n)
    for (train_idx, test_idx) in folds
        X_tr = X[train_idx, :]
        y_tr = y[train_idx]
        w_tr = weights === nothing ? nothing : weights[train_idx]
        
        fitted_fold = fit(model.estimator, X_tr, y_tr; weights=w_tr, context=context)
        
        base_classes = fitted_fold.classes
        base_probs = predict_proba(fitted_fold, X[test_idx, :])
        col_2 = findfirst(==(classes[2]), base_classes)
        if col_2 !== nothing
            oof_p[test_idx] = base_probs[:, col_2]
        else
            oof_p[test_idx] .= 0.0
        end
    end
    
    y_true_bin = [val == classes[2] ? 1.0 : 0.0 for val in y]
    best_threshold = 0.5
    best_score = -1.0
    
    thresholds = 0.0:0.01:1.0
    for t in thresholds
        y_pred_bin = [p >= t ? 1.0 : 0.0 for p in oof_p]
        score = _evaluate_threshold_metric(model.metric, y_true_bin, y_pred_bin)
        if score > best_score
            best_score = score
            best_threshold = t
        end
    end
    
    fitted_estimator = fit(model.estimator, X, y; weights=weights, context=context)
    
    details = (threshold=best_threshold, best_score=best_score, classes=copy(classes))
    fit_report = FitReport(status=:success, observations=n, features=size(X,2),
                           backend=:cpu, details=details, context=context)
                           
    FittedThresholdSelectionWrapper(model, fitted_estimator, best_threshold, classes, fit_report,
                                    with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedThresholdSelectionWrapper, X::AbstractMatrix)
    predict_proba(fitted.fitted_estimator, X)
end

function predict(fitted::FittedThresholdSelectionWrapper, X::AbstractMatrix)
    probs_raw = predict_proba(fitted.fitted_estimator, X)
    base_classes = fitted.fitted_estimator.classes
    col_2 = findfirst(==(fitted.classes[2]), base_classes)
    
    n = size(X, 1)
    preds = Vector{eltype(fitted.classes)}(undef, n)
    for row in 1:n
        p = col_2 !== nothing ? probs_raw[row, col_2] : 0.0
        preds[row] = p >= fitted.threshold ? fitted.classes[2] : fitted.classes[1]
    end
    return preds
end

report(fitted::FittedThresholdSelectionWrapper) = fitted.report

