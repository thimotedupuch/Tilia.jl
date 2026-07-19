# ----------------- ClassifierChain -----------------

struct ClassifierChain{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    order::Union{Vector{Int}, Nothing}
    function ClassifierChain(estimator::AbstractPredictor; order=nothing)
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("ClassifierChain requires a classification base estimator."))
        new{typeof(estimator)}(estimator, order)
    end
end

struct FittedClassifierChain{M,F,O,R,S} <: AbstractFittedEstimator
    model::M
    estimators::F
    order::O
    report::R
    schema::S
end

capabilities(::Type{<:ClassifierChain}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::ClassifierChain, X::AbstractMatrix, y::AbstractMatrix;
             weights=nothing, context=default_context())
    n, m = size(y)
    order = model.order === nothing ? collect(1:m) : model.order
    length(order) == m || throw(SchemaMismatchError("ClassifierChain order length must match target column count."))
    
    T = eltype(X)
    estimators = []
    
    for j in 1:m
        col = order[j]
        y_col = view(y, :, col)
        
        if j == 1
            X_input = X
        else
            prev_cols = [order[k] for k in 1:j-1]
            prev_y = Matrix{T}(y[:, prev_cols])
            X_input = hcat(X, prev_y)
        end
        
        fitted = fit(model.estimator, X_input, y_col; weights=weights, context=context)
        push!(estimators, fitted)
    end
    
    details = (order=order,)
    fit_report = FitReport(status=:success, observations=n, features=size(X,2),
                           backend=:cpu, details=details, context=context)
                           
    FittedClassifierChain(model, estimators, order, fit_report,
                          with_class_target(infer_schema(X), unique(y)))
end

function predict(fitted::FittedClassifierChain, X::AbstractMatrix)
    n = size(X, 1)
    m = length(fitted.estimators)
    T = eltype(X)
    
    preds = Matrix{Any}(undef, n, m)
    for j in 1:m
        col = fitted.order[j]
        if j == 1
            X_input = X
        else
            prev_cols = [fitted.order[k] for k in 1:j-1]
            prev_y = Matrix{T}(preds[:, prev_cols])
            X_input = hcat(X, prev_y)
        end
        preds[:, col] = predict(fitted.estimators[j], X_input)
    end
    return preds
end

function predict_proba(fitted::FittedClassifierChain, X::AbstractMatrix)
    n = size(X, 1)
    m = length(fitted.estimators)
    T = eltype(X)
    
    preds = Matrix{Any}(undef, n, m)
    probs = Vector{Any}(undef, m)
    
    for j in 1:m
        col = fitted.order[j]
        if j == 1
            X_input = X
        else
            prev_cols = [fitted.order[k] for k in 1:j-1]
            prev_y = Matrix{T}(preds[:, prev_cols])
            X_input = hcat(X, prev_y)
        end
        probs[col] = predict_proba(fitted.estimators[j], X_input)
        preds[:, col] = predict(fitted.estimators[j], X_input)
    end
    return probs
end

report(fitted::FittedClassifierChain) = fitted.report

