# ----------------- BaggingRegressor -----------------

struct BaggingRegressor{E<:AbstractPredictor} <: AbstractPredictor
    base_estimator::E
    n_estimators::Int
    max_samples::Float64
    max_features::Float64
    bootstrap::Bool
    function BaggingRegressor(base_estimator::AbstractPredictor;
                              n_estimators::Integer=10, max_samples::Real=1.0,
                              max_features::Real=1.0, bootstrap::Bool=true)
        n_estimators > 0 || throw(InvalidHyperparameterError("BaggingRegressor n_estimators must be positive."))
        0.0 < max_samples <= 1.0 || throw(InvalidHyperparameterError("BaggingRegressor max_samples must be in (0, 1]."))
        0.0 < max_features <= 1.0 || throw(InvalidHyperparameterError("BaggingRegressor max_features must be in (0, 1]."))
        new{typeof(base_estimator)}(base_estimator, Int(n_estimators), Float64(max_samples), Float64(max_features), bootstrap)
    end
end

struct FittedBaggingRegressor{M,F,Feat,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    features::Feat
    report::R
    schema::S
end

capabilities(::Type{<:BaggingRegressor}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function fit(model::BaggingRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    n, d = size(X)
    estimators = []
    features_list = []
    
    n_samples_draw = round(Int, model.max_samples * n)
    n_features_draw = round(Int, model.max_features * d)
    n_samples_draw = max(1, n_samples_draw)
    n_features_draw = max(1, n_features_draw)
    
    for i in 1:model.n_estimators
        obs_idx = model.bootstrap ? rand(context.rng, 1:n, n_samples_draw) :
                                    randperm(context.rng, n)[1:n_samples_draw]
        
        feat_idx = sort!(randperm(context.rng, d)[1:n_features_draw])
        
        X_sub = X[obs_idx, feat_idx]
        y_sub = y[obs_idx]
        w_sub = weights === nothing ? nothing : weights[obs_idx]
        
        fitted = fit(model.base_estimator, X_sub, y_sub; weights=w_sub, context=context)
        push!(estimators, fitted)
        push!(features_list, feat_idx)
    end
    
    details = (n_estimators=model.n_estimators,)
    fit_report = FitReport(status=:success, observations=n, features=d,
                           backend=:cpu, details=details, context=context)
                           
    FittedBaggingRegressor(model, estimators, features_list, fit_report,
                           with_target(infer_schema(X), y))
end

function predict(fitted::FittedBaggingRegressor, X::AbstractMatrix)
    n = size(X, 1)
    preds = zeros(Float64, n)
    for i in 1:fitted.model.n_estimators
        feat_idx = fitted.features[i]
        preds .+= predict(fitted.estimators[i], X[:, feat_idx])
    end
    preds ./= fitted.model.n_estimators
    return preds
end

report(fitted::FittedBaggingRegressor) = fitted.report

