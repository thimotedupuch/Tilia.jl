# ----------------- OneVsOneClassifier -----------------

struct OneVsOneClassifier{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    function OneVsOneClassifier(estimator::AbstractPredictor)
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("OneVsOneClassifier requires a classification base estimator."))
        new{typeof(estimator)}(estimator)
    end
end

struct FittedOneVsOneClassifier{M,F,P,L,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    pairs::P
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:OneVsOneClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::OneVsOneClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    K = length(classes)
    
    estimators = []
    pairs = []
    
    for i in 1:K
        for j in i+1:K
            c_i = classes[i]
            c_j = classes[j]
            idx = findall(val -> val == c_i || val == c_j, y)
            if isempty(idx)
                continue
            end
            X_pair = X[idx, :]
            y_pair = [val == c_j ? 1 : 0 for val in y[idx]]
            w_pair = weights === nothing ? nothing : weights[idx]
            
            fitted = fit(model.estimator, X_pair, y_pair; weights=w_pair, context=context)
            push!(estimators, fitted)
            push!(pairs, (c_i, c_j))
        end
    end
    
    details = (classes=copy(classes),)
    fit_report = FitReport(status=:success, observations=size(X,1), features=size(X,2),
                           backend=:cpu, details=details, context=context)
    
    FittedOneVsOneClassifier(model, estimators, pairs, classes, fit_report,
                             with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedOneVsOneClassifier, X::AbstractMatrix)
    n = size(X, 1)
    K = length(fitted.classes)
    votes = zeros(Float64, n, K)
    class_to_idx = Dict(c => idx for (idx, c) in enumerate(fitted.classes))
    
    for (m, (c_i, c_j)) in enumerate(fitted.pairs)
        idx_i = class_to_idx[c_i]
        idx_j = class_to_idx[c_j]
        preds = predict(fitted.estimators[m], X)
        for row in 1:n
            if preds[row] == 1
                votes[row, idx_j] += 1.0
            else
                votes[row, idx_i] += 1.0
            end
        end
    end
    
    for row in 1:n
        s = sum(votes[row, :])
        if s > 0
            votes[row, :] ./= s
        else
            votes[row, :] .= 1.0 / K
        end
    end
    return votes
end

function predict(fitted::FittedOneVsOneClassifier, X::AbstractMatrix)
    probs = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
end

report(fitted::FittedOneVsOneClassifier) = fitted.report

