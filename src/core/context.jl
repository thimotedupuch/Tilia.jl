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
end
CompilationCache() = CompilationCache(Dict{UInt64,Any}(), ReentrantLock())

"""Numerical defaults shared by all estimators."""
struct NumericsPolicy{T<:AbstractFloat}
    float_type::Type{T}
    tolerance::T
end
NumericsPolicy(::Type{T}=Float64; tolerance=sqrt(eps(T))) where {T<:AbstractFloat} =
    NumericsPolicy{T}(T, T(tolerance))

"""Execution configuration supplied to `fit`."""
struct FitContext{B<:AbstractBackend,R<:AbstractRNG,N<:NumericsPolicy,C<:CompilationCache}
    backend::B
    rng::R
    numerics::N
    deterministic::Bool
    cache::C
end

FitContext(; backend=CPUBackend(), rng=Random.Xoshiro(0),
           numerics=NumericsPolicy(), deterministic=true,
           cache=CompilationCache()) =
    FitContext(backend, rng, numerics, deterministic, cache)

default_context() = FitContext()

function require_cpu(context::FitContext, operation::AbstractString)
    context.backend isa CPUBackend || throw(UnsupportedBackendError(
        "$operation is currently implemented only for CPUBackend; use FitContext(backend=CPUBackend()) or install a supported backend extension."))
    context
end
