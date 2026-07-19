# ----------------- OneVsRestClassifier -----------------

struct OneVsRestClassifier{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    function OneVsRestClassifier(estimator::AbstractPredictor)
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("OneVsRestClassifier requires a classification base estimator."))
        new{typeof(estimator)}(estimator)
    end
end

struct FittedOneVsRestClassifier{M,F,L,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:OneVsRestClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::OneVsRestClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    K = length(classes)
    
    estimators = []
    classes_to_fit = K == 2 ? classes[2:2] : classes
    for c in classes_to_fit
        y_bin = [val == c ? 1 : 0 for val in y]
        fitted = fit(model.estimator, X, y_bin; weights=weights, context=context)
        push!(estimators, fitted)
    end
    
    details = (classes=copy(classes),)
    fit_report = FitReport(status=:success, observations=size(X,1), features=size(X,2),
                           backend=:cpu, details=details, context=context)
    
    FittedOneVsRestClassifier(model, estimators, classes, fit_report,
                              with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedOneVsRestClassifier, X::AbstractMatrix)
    n = size(X, 1)
    K = length(fitted.classes)
    probs = Matrix{Float64}(undef, n, K)
    
    if K == 2
        p = predict_proba(fitted.estimators[1], X)[:, 2]
        probs[:, 1] = 1.0 .- p
        probs[:, 2] = p
    else
        for k in 1:K
            probs[:, k] = predict_proba(fitted.estimators[k], X)[:, 2]
        end
        for i in 1:n
            s = sum(probs[i, :])
            if s > 0
                probs[i, :] ./= s
            else
                probs[i, :] .= 1.0 / K
            end
        end
    end
    return probs
end

function predict(fitted::FittedOneVsRestClassifier, X::AbstractMatrix)
    probs = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
end

report(fitted::FittedOneVsRestClassifier) = fitted.report

