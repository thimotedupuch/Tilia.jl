"""Machine-readable diagnostics produced while fitting an estimator."""
struct FitReport
    status::Symbol
    observations::Int
    features::Int
    backend::Symbol
    warnings::Vector{String}
    details::NamedTuple
end

FitReport(; status=:success, observations=0, features=0, backend=:cpu,
          warnings=String[], details=(;)) =
    FitReport(status, observations, features, backend, warnings, details)

struct ConfusionMatrix{T,L}; matrix::Matrix{T}; labels::Vector{L}; end
struct ROCResult{T}; false_positive_rate::Vector{T}; true_positive_rate::Vector{T}; thresholds::Vector{T}; end
struct CrossValidationResult{T,M,R}
    scores::Vector{T}
    fitted_models::Vector{M}
    fold_reports::Vector{R}
    train_indices::Vector{Vector{Int}}
    test_indices::Vector{Vector{Int}}
end
struct OptimizationTrace{T}; objective::Vector{T}; converged::Bool; end

"""Structured result of deterministic hyperparameter search."""
struct TuningResult{M,P,T,F}
    best_model::M
    best_parameters::P
    best_score::T
    trials::Vector{NamedTuple}
    fitted_model::F
end
