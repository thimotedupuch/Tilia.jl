# ----------------- MultiOutputClassifier -----------------

struct MultiOutputClassifier{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    function MultiOutputClassifier(estimator::AbstractPredictor)
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("MultiOutputClassifier requires a classification base estimator."))
        new{typeof(estimator)}(estimator)
    end
end

struct FittedMultiOutputClassifier{M,F,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    report::R
    schema::S
end

capabilities(::Type{<:MultiOutputClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::MultiOutputClassifier, X::AbstractMatrix, y::AbstractMatrix;
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
    
    FittedMultiOutputClassifier(model, estimators, fit_report,
                                with_class_target(infer_schema(X), unique(y)))
end

function predict(fitted::FittedMultiOutputClassifier, X::AbstractMatrix)
    n = size(X, 1)
    m = length(fitted.estimators)
    sample_pred = predict(fitted.estimators[1], X)
    preds = Matrix{eltype(sample_pred)}(undef, n, m)
    preds[:, 1] = sample_pred
    for col in 2:m
        preds[:, col] = predict(fitted.estimators[col], X)
    end
    return preds
end

function predict_proba(fitted::FittedMultiOutputClassifier, X::AbstractMatrix)
    return [predict_proba(est, X) for est in fitted.estimators]
end

report(fitted::FittedMultiOutputClassifier) = fitted.report

