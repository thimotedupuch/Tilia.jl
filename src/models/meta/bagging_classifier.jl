# ----------------- BaggingClassifier -----------------

struct BaggingClassifier{E<:AbstractPredictor} <: AbstractPredictor
    base_estimator::E
    n_estimators::Int
    max_samples::Float64
    max_features::Float64
    bootstrap::Bool
    function BaggingClassifier(base_estimator::AbstractPredictor;
                               n_estimators::Integer=10, max_samples::Real=1.0,
                               max_features::Real=1.0, bootstrap::Bool=true)
        n_estimators > 0 || throw(InvalidHyperparameterError("BaggingClassifier n_estimators must be positive."))
        0.0 < max_samples <= 1.0 || throw(InvalidHyperparameterError("BaggingClassifier max_samples must be in (0, 1]."))
        0.0 < max_features <= 1.0 || throw(InvalidHyperparameterError("BaggingClassifier max_features must be in (0, 1]."))
        new{typeof(base_estimator)}(base_estimator, Int(n_estimators), Float64(max_samples), Float64(max_features), bootstrap)
    end
end

struct FittedBaggingClassifier{M,F,Feat,L,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    features::Feat
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:BaggingClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::BaggingClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    K = length(classes)
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
    
    details = (n_estimators=model.n_estimators, classes=copy(classes))
    fit_report = FitReport(status=:success, observations=n, features=d,
                           backend=:cpu, details=details, context=context)
                           
    FittedBaggingClassifier(model, estimators, features_list, classes, fit_report,
                            with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedBaggingClassifier, X::AbstractMatrix)
    n = size(X, 1)
    K = length(fitted.classes)
    probs = zeros(Float64, n, K)
    
    for i in 1:fitted.model.n_estimators
        feat_idx = fitted.features[i]
        base_classes = fitted.estimators[i].classes
        base_probs = predict_proba(fitted.estimators[i], X[:, feat_idx])
        
        for (col_idx, cls) in enumerate(base_classes)
            target_idx = findfirst(==(cls), fitted.classes)
            if target_idx !== nothing
                probs[:, target_idx] .+= base_probs[:, col_idx]
            end
        end
    end
    probs ./= fitted.model.n_estimators
    return probs
end

function predict(fitted::FittedBaggingClassifier, X::AbstractMatrix)
    probs = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
end

report(fitted::FittedBaggingClassifier) = fitted.report

