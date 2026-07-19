"""
Compositional and meta-estimators: OneVsRest, OneVsOne, MultiOutput, Stacking, Voting,
ClassifierChain, Bagging, TransformedTargetRegressor, CalibratedClassifier, and ThresholdSelectionWrapper.
"""


# Platt scaling objective for CalibratedClassifier
function platt_objective(θ, p, y_bin, weights)
    T = eltype(θ)
    A, B = θ[1], θ[2]
    loss = zero(T)
    for i in eachindex(p)
        w = weights === nothing ? one(T) : T(weights[i])
        sig = Kernels.sigmoid(A * p[i] + B)
        sig = clamp(sig, T(1e-15), one(T) - T(1e-15))
        loss -= w * (y_bin[i] * log(sig) + (1.0 - y_bin[i]) * log(1.0 - sig))
    end
    return loss
end

function platt_gradient!(grad, θ, p, y_bin, weights)
    T = eltype(θ)
    A, B = θ[1], θ[2]
    grad[1] = zero(T)
    grad[2] = zero(T)
    for i in eachindex(p)
        w = weights === nothing ? one(T) : T(weights[i])
        sig = Kernels.sigmoid(A * p[i] + B)
        diff = sig - y_bin[i]
        grad[1] += w * diff * p[i]
        grad[2] += w * diff
    end
    return grad
end

# Threshold evaluation helper
function _evaluate_threshold_metric(metric, y_true, y_pred_bin)
    tp = sum((y_true .== 1.0) .& (y_pred_bin .== 1.0))
    fp = sum((y_true .== 0.0) .& (y_pred_bin .== 1.0))
    fn = sum((y_true .== 1.0) .& (y_pred_bin .== 0.0))
    tn = sum((y_true .== 0.0) .& (y_pred_bin .== 0.0))
    
    if metric === :accuracy
        return (tp + tn) / max(1, length(y_true))
    elseif metric === :f1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0.0
            return 0.0
        end
        return 2.0 * precision * recall / (precision + recall)
    elseif metric === :balanced_accuracy
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        return 0.5 * (tpr + tnr)
    end
    return 0.0
end

