abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct ReactantBackend <: AbstractBackend
    fallback::Symbol
    device::Symbol
    function ReactantBackend(; fallback::Symbol=:error, device::Symbol=:auto)
        fallback in (:error, :cpu) || throw(InvalidHyperparameterError(
            "ReactantBackend fallback must be :error or :cpu."))
        device in (:auto, :cpu, :gpu) || throw(InvalidHyperparameterError(
            "ReactantBackend device must be :auto, :cpu, or :gpu."))
        new(fallback, device)
    end
end

"""Thread-safe, context-owned cache for optional backend compilations."""
mutable struct CompilationCache
    entries::Dict{UInt64,Any}
    lock::ReentrantLock
    order::Vector{UInt64}
    max_entries::Int
    compilations::Int
    evictions::Int
end
function CompilationCache(; max_entries::Integer=32)
    max_entries > 0 || throw(InvalidHyperparameterError(
        "CompilationCache max_entries must be positive."))
    CompilationCache(Dict{UInt64,Any}(), ReentrantLock(), UInt64[],
                     Int(max_entries), 0, 0)
end
CompilationCache(entries::Dict{UInt64,Any}, lock::ReentrantLock) =
    CompilationCache(entries, lock, collect(keys(entries)), max(length(entries), 32),
                     length(entries), 0)

"""Release all compiled entries retained by a compilation cache."""
function Base.empty!(cache::CompilationCache)
    lock(cache.lock) do
        empty!(cache.entries)
        empty!(cache.order)
    end
    cache
end

function _compilation_cache_snapshot(cache::CompilationCache)
    lock(cache.lock) do
        (size=length(cache.entries), capacity=cache.max_entries,
         compilations=cache.compilations, evictions=cache.evictions,
         retained_host_bytes=Base.summarysize(cache.entries) +
                             Base.summarysize(cache.order))
    end
end

"""Numerical defaults shared by all estimators."""
struct NumericsPolicy{T<:AbstractFloat,A<:AbstractFloat}
    float_type::Type{T}
    accumulation_type::Type{A}
    tolerance::T
    tolerance_scale::T
    max_iterations::Int
    stable_summation::Bool
    missing_policy::Symbol
    finite_policy::Symbol
    overflow_policy::Symbol
    underflow_policy::Symbol
    deterministic_reductions::Bool
    sparse_centering::Symbol
end

function NumericsPolicy(::Type{T}=Float64; accumulation_type::Type{A}=T,
                        tolerance=sqrt(eps(T)), tolerance_scale=one(T),
                        max_iterations::Integer=1_000,
                        stable_summation::Bool=true,
                        missing_policy::Symbol=:error,
                        finite_policy::Symbol=:error,
                        overflow_policy::Symbol=:error,
                        underflow_policy::Symbol=:allow,
                        deterministic_reductions::Bool=true,
                        sparse_centering::Symbol=:error) where
                        {T<:AbstractFloat,A<:AbstractFloat}
    isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
        "NumericsPolicy tolerance must be finite and positive."))
    isfinite(tolerance_scale) && tolerance_scale > 0 || throw(InvalidHyperparameterError(
        "NumericsPolicy tolerance_scale must be finite and positive."))
    max_iterations > 0 || throw(InvalidHyperparameterError(
        "NumericsPolicy max_iterations must be positive."))
    missing_policy in (:error, :allow) || throw(InvalidHyperparameterError(
        "NumericsPolicy missing_policy must be :error or :allow."))
    finite_policy in (:error, :allow) || throw(InvalidHyperparameterError(
        "NumericsPolicy finite_policy must be :error or :allow."))
    overflow_policy in (:error, :saturate) || throw(InvalidHyperparameterError(
        "NumericsPolicy overflow_policy must be :error or :saturate."))
    underflow_policy in (:allow, :flush_zero, :error) || throw(InvalidHyperparameterError(
        "NumericsPolicy underflow_policy must be :allow, :flush_zero, or :error."))
    sparse_centering in (:error, :densify) || throw(InvalidHyperparameterError(
        "NumericsPolicy sparse_centering must be :error or :densify."))
    NumericsPolicy{T,A}(T, A, T(tolerance), T(tolerance_scale),
        Int(max_iterations), stable_summation, missing_policy, finite_policy,
        overflow_policy, underflow_policy, deterministic_reductions, sparse_centering)
end

numerics_summary(policy::NumericsPolicy) = (
    float_type=string(policy.float_type),
    accumulation_type=string(policy.accumulation_type),
    tolerance=policy.tolerance,
    tolerance_scale=policy.tolerance_scale,
    max_iterations=policy.max_iterations,
    stable_summation=policy.stable_summation,
    missing_policy=policy.missing_policy,
    finite_policy=policy.finite_policy,
    overflow_policy=policy.overflow_policy,
    underflow_policy=policy.underflow_policy,
    deterministic_reductions=policy.deterministic_reductions,
    sparse_centering=policy.sparse_centering,
)

"""Scale an estimator tolerance according to the active numerical policy."""
effective_tolerance(context, requested=context.numerics.tolerance) =
    requested * context.numerics.tolerance_scale

"""Apply the active numerical policy's hard iteration ceiling."""
effective_max_iterations(context, requested::Integer=context.numerics.max_iterations) =
    min(Int(requested), context.numerics.max_iterations)

"""Execution configuration supplied to `fit`."""
struct FitContext{B<:AbstractBackend,R<:AbstractRNG,N<:NumericsPolicy,C<:CompilationCache}
    backend::B
    rng::R
    numerics::N
    deterministic::Bool
    cache::C
    root_seed::UInt64
    stream_id::String
end

function FitContext(; backend=CPUBackend(), seed::Integer=0, rng=nothing,
           numerics=NumericsPolicy(), deterministic=true,
           cache=CompilationCache(), root_seed::Integer=seed,
           stream_id::AbstractString="root")
    root = UInt64(root_seed)
    generator = rng === nothing ? Random.Xoshiro(root) : rng
    FitContext(backend, generator, numerics, deterministic, cache, root, String(stream_id))
end

default_context() = FitContext()

function _derived_seed(root_seed::UInt64, stream_id::AbstractString, components)
    description = join((string(root_seed), stream_id, map(repr, components)...), '|')
    digest = sha256(codeunits(description))
    seed = zero(UInt64)
    for byte in digest[1:8]
        seed = (seed << 8) | UInt64(byte)
    end
    seed
end

"""Derive a deterministic named random substream without mutating its parent."""
function derive_context(context::FitContext, components...)
    suffix = join(map(string, components), '/')
    identifier = isempty(context.stream_id) ? suffix : string(context.stream_id, '/', suffix)
    seed = _derived_seed(context.root_seed, context.stream_id, components)
    FitContext(backend=context.backend, rng=Random.Xoshiro(seed),
        numerics=context.numerics, deterministic=context.deterministic,
        cache=context.cache, root_seed=context.root_seed, stream_id=identifier)
end

function require_cpu(context::FitContext, operation::AbstractString)
    context.backend isa CPUBackend || throw(UnsupportedBackendError(
        "$operation is currently implemented only for CPUBackend; use FitContext(backend=CPUBackend()) or install a supported backend extension."))
    context
end
