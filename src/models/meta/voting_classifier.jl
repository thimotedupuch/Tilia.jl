# ----------------- VotingClassifier -----------------

struct VotingClassifier{E<:Tuple} <: AbstractPredictor
    estimators::E
    voting::Symbol
    weights::Union{Vector{Float64}, Nothing}
    function VotingClassifier(estimators::AbstractPredictor...; voting::Symbol=:soft, weights=nothing)
        isempty(estimators) && throw(InvalidHyperparameterError("VotingClassifier requires at least one estimator."))
        all(est -> capabilities(typeof(est)).task === :classification, estimators) ||
            throw(InvalidHyperparameterError("VotingClassifier estimators must all be classification models."))
        voting in (:hard, :soft) || throw(InvalidHyperparameterError("VotingClassifier voting must be :hard or :soft."))
        weights !== nothing && length(weights) != length(estimators) &&
            throw(InvalidHyperparameterError("VotingClassifier weights length must match estimators count."))
        new{typeof(estimators)}(estimators, voting, weights === nothing ? nothing : Float64.(weights))
    end
end

struct FittedVotingClassifier{M,F,L,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:VotingClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::VotingClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    fitted_estimators = [fit(est, X, y; weights=weights, context=context) for est in model.estimators]
    
    details = (num_estimators=length(fitted_estimators), classes=copy(classes))
    fit_report = FitReport(status=:success, observations=size(X,1), features=size(X,2),
                           backend=:cpu, details=details, context=context)
                           
    FittedVotingClassifier(model, fitted_estimators, classes, fit_report,
                           with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedVotingClassifier, X::AbstractMatrix)
    fitted.model.voting === :soft || throw(UnsupportedDataError("predict_proba is only supported for soft voting."))
    n = size(X, 1)
    K = length(fitted.classes)
    probs = zeros(Float64, n, K)
    w_sum = fitted.model.weights === nothing ? Float64(length(fitted.estimators)) : sum(fitted.model.weights)
    
    for (idx, est) in enumerate(fitted.estimators)
        w = fitted.model.weights === nothing ? 1.0 : fitted.model.weights[idx]
        base_classes = est.classes
        base_probs = predict_proba(est, X)
        for (col_idx, cls) in enumerate(base_classes)
            target_idx = findfirst(==(cls), fitted.classes)
            if target_idx !== nothing
                probs[:, target_idx] .+= w .* base_probs[:, col_idx]
            end
        end
    end
    probs ./= w_sum
    return probs
end

function predict(fitted::FittedVotingClassifier, X::AbstractMatrix)
    n = size(X, 1)
    if fitted.model.voting === :soft
        probs = predict_proba(fitted, X)
        return [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
    else
        K = length(fitted.classes)
        votes = zeros(Float64, n, K)
        class_to_idx = Dict(c => idx for (idx, c) in enumerate(fitted.classes))
        
        for (idx, est) in enumerate(fitted.estimators)
            w = fitted.model.weights === nothing ? 1.0 : fitted.model.weights[idx]
            preds = predict(est, X)
            for row in 1:n
                target_idx = class_to_idx[preds[row]]
                votes[row, target_idx] += w
            end
        end
        return [fitted.classes[argmax(view(votes, row, :))] for row in axes(votes, 1)]
    end
end

report(fitted::FittedVotingClassifier) = fitted.report

