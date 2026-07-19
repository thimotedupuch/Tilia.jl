# ----------------- VotingRegressor -----------------

struct VotingRegressor{E<:Tuple} <: AbstractPredictor
    estimators::E
    weights::Union{Vector{Float64}, Nothing}
    function VotingRegressor(estimators::AbstractPredictor...; weights=nothing)
        isempty(estimators) && throw(InvalidHyperparameterError("VotingRegressor requires at least one estimator."))
        all(est -> capabilities(typeof(est)).task === :regression, estimators) ||
            throw(InvalidHyperparameterError("VotingRegressor estimators must all be regression models."))
        weights !== nothing && length(weights) != length(estimators) &&
            throw(InvalidHyperparameterError("VotingRegressor weights length must match estimators count."))
        new{typeof(estimators)}(estimators, weights === nothing ? nothing : Float64.(weights))
    end
end

struct FittedVotingRegressor{M,F,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    report::R
    schema::S
end

capabilities(::Type{<:VotingRegressor}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::VotingRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    fitted_estimators = [fit(est, X, y; weights=weights, context=context) for est in model.estimators]
    
    details = (num_estimators=length(fitted_estimators),)
    fit_report = FitReport(status=:success, observations=size(X,1), features=size(X,2),
                           backend=:cpu, details=details, context=context)
                           
    FittedVotingRegressor(model, fitted_estimators, fit_report,
                          with_target(infer_schema(X), y))
end

function predict(fitted::FittedVotingRegressor, X::AbstractMatrix)
    n = size(X, 1)
    preds = zeros(Float64, n)
    w_sum = fitted.model.weights === nothing ? Float64(length(fitted.estimators)) : sum(fitted.model.weights)
    
    for (idx, est) in enumerate(fitted.estimators)
        w = fitted.model.weights === nothing ? 1.0 : fitted.model.weights[idx]
        preds .+= w .* predict(est, X)
    end
    preds ./= w_sum
    return preds
end

report(fitted::FittedVotingRegressor) = fitted.report

