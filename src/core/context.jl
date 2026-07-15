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
