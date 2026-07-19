# ----------------- CalibratedClassifier -----------------

struct CalibratedClassifier{E<:AbstractPredictor} <: AbstractPredictor
    estimator::E
    cv::KFold
    function CalibratedClassifier(estimator::AbstractPredictor; cv=KFold(5))
        capabilities(typeof(estimator)).task === :classification || throw(InvalidHyperparameterError("CalibratedClassifier requires a classification base estimator."))
        new{typeof(estimator)}(estimator, cv)
    end
end

struct FittedCalibratedClassifier{M,F,A,B,L,R,S} <: AbstractFittedEstimator
    model::M
    fitted_estimator::F
    Platt_A::A
    Platt_B::B
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:CalibratedClassifier}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::CalibratedClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    classes = _classification_classes(y)
    K = length(classes)
    n = size(X, 1)
    
    folds = split(model.cv, n)
    oof_probs = Matrix{Float64}(undef, n, K)
    
    for (train_idx, test_idx) in folds
        X_tr = X[train_idx, :]
        y_tr = y[train_idx]
        w_tr = weights === nothing ? nothing : weights[train_idx]
        
        fitted_fold = fit(model.estimator, X_tr, y_tr; weights=w_tr, context=context)
        
        base_classes = fitted_fold.classes
        base_probs = predict_proba(fitted_fold, X[test_idx, :])
        
        fill!(view(oof_probs, test_idx, :), 0.0)
        for (col_idx, cls) in enumerate(base_classes)
            target_idx = findfirst(==(cls), classes)
            if target_idx !== nothing
                oof_probs[test_idx, target_idx] = base_probs[:, col_idx]
            end
        end
    end
    
    Platt_A = zeros(Float64, K)
    Platt_B = zeros(Float64, K)
    
    classes_to_calibrate = K == 2 ? (2:2) : (1:K)
    for c_idx in classes_to_calibrate
        y_bin = [val == classes[c_idx] ? 1.0 : 0.0 for val in y]
        p = view(oof_probs, :, c_idx)
        
        initial = [1.0, 0.0]
        obj = θ -> platt_objective(θ, p, y_bin, weights)
        grad! = (g, θ) -> platt_gradient!(g, θ, p, y_bin, weights)
        
        result = Solvers.lbfgs(obj, grad!, initial; tolerance=1e-6, max_iterations=200)
        Platt_A[c_idx] = result.parameters[1]
        Platt_B[c_idx] = result.parameters[2]
    end
    
    fitted_estimator = fit(model.estimator, X, y; weights=weights, context=context)
    
    details = (classes=copy(classes), Platt_A=Platt_A, Platt_B=Platt_B)
    fit_report = FitReport(status=:success, observations=n, features=size(X,2),
                           backend=:cpu, details=details, context=context)
                           
    FittedCalibratedClassifier(model, fitted_estimator, Platt_A, Platt_B, classes, fit_report,
                               with_class_target(infer_schema(X), classes))
end

function predict_proba(fitted::FittedCalibratedClassifier, X::AbstractMatrix)
    n = size(X, 1)
    K = length(fitted.classes)
    
    base_probs_raw = predict_proba(fitted.fitted_estimator, X)
    base_classes = fitted.fitted_estimator.classes
    
    base_probs = zeros(Float64, n, K)
    for (col_idx, cls) in enumerate(base_classes)
        target_idx = findfirst(==(cls), fitted.classes)
        if target_idx !== nothing
            base_probs[:, target_idx] = base_probs_raw[:, col_idx]
        end
    end
    
    probs = zeros(Float64, n, K)
    if K == 2
        p_cal = Kernels.sigmoid.(fitted.Platt_A[2] .* base_probs[:, 2] .+ fitted.Platt_B[2])
        probs[:, 1] = 1.0 .- p_cal
        probs[:, 2] = p_cal
    else
        for c_idx in 1:K
            probs[:, c_idx] = Kernels.sigmoid.(fitted.Platt_A[c_idx] .* base_probs[:, c_idx] .+ fitted.Platt_B[c_idx])
        end
        for row in 1:n
            s = sum(probs[row, :])
            if s > 0
                probs[row, :] ./= s
            else
                probs[row, :] .= 1.0 / K
            end
        end
    end
    return probs
end

function predict(fitted::FittedCalibratedClassifier, X::AbstractMatrix)
    probs = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probs, row, :))] for row in axes(probs, 1)]
end

report(fitted::FittedCalibratedClassifier) = fitted.report

