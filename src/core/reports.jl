"""Machine-readable diagnostics produced while fitting an estimator."""
struct FitReport
    status::Symbol
    observations::Int
    features::Int
    backend::Symbol
    warnings::Vector{String}
    details::NamedTuple
    root_seed::UInt64
    stream_id::String
    deterministic::Bool
    thread_count::Int
end

function FitReport(; status=:success, observations=0, features=0, backend=:cpu,
                   warnings=String[], details=(;), context=nothing,
                   root_seed=nothing, stream_id=nothing, deterministic=nothing,
                   thread_count=nothing)
    enriched_details = if context === nothing || hasproperty(details, :numerical_policy)
        details
    else
        merge(details, (numerical_policy=numerics_summary(context.numerics),))
    end
    FitReport(status, observations, features, backend, warnings, enriched_details,
        root_seed === nothing ?
            (context === nothing ? UInt64(0) : context.root_seed) : UInt64(root_seed),
        stream_id === nothing ?
            (context === nothing ? "unknown" : context.stream_id) : String(stream_id),
        deterministic === nothing ?
            (context === nothing ? true : context.deterministic) : Bool(deterministic),
        thread_count === nothing ? Threads.nthreads() : Int(thread_count))
end

# Positional compatibility for version-1 structural persistence payloads.
FitReport(status::Symbol, observations::Int, features::Int, backend::Symbol,
          warnings::Vector{String}, details::NamedTuple) =
    FitReport(status, observations, features, backend, warnings, details,
              UInt64(0), "migrated", true, 1)

struct ConfusionMatrix{T,L}; matrix::Matrix{T}; labels::Vector{L}; end
struct ROCResult{T}; false_positive_rate::Vector{T}; true_positive_rate::Vector{T}; thresholds::Vector{T}; end
struct PrecisionRecallResult{T}; precision::Vector{T}; recall::Vector{T}; thresholds::Vector{T}; end
struct CalibrationResult{T}
    mean_predicted_probability::Vector{T}
    fraction_positive::Vector{T}
    counts::Vector{T}
    bin_edges::Vector{T}
end
struct PermutationImportanceResult{T,L}
    baseline_score::T
    importances::Matrix{T}
    mean_importance::Vector{T}
    standard_deviation::Vector{T}
    feature_names::Vector{L}
end
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
