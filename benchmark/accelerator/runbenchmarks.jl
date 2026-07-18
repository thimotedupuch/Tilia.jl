using Random
using Statistics
using Tilia
using Reactant

rng = Xoshiro(42)
observation_sizes = haskey(ENV, "TILIA_ACCELERATOR_SIZES") ?
    Tuple(parse.(Int, split(ENV["TILIA_ACCELERATOR_SIZES"], ','))) :
    (100, 1_000, 10_000)
fit_samples = parse(Int, get(ENV, "TILIA_ACCELERATOR_FIT_SAMPLES", "3"))

for observations in observation_sizes
    X = randn(rng, Float32, observations, 16)
    y = [row <= observations ÷ 2 ? :negative : :positive
         for row in axes(X, 1)]
    model = Chain(Standardize(), LogisticRegression(lambda=1.0f0))

    cpu_fit = @elapsed cpu = fit(model, X, y)
    cpu_fit_steady = median([@elapsed fit(model, X, y) for _ in 1:fit_samples])
    cpu_operation = () -> predict_proba(cpu, X)
    cpu_first = @elapsed cpu_operation()
    cpu_steady = median([@elapsed cpu_operation() for _ in 1:3])

    context = FitContext(backend=ReactantBackend(device=:cpu),
                         cache=CompilationCache())
    accelerated_fit = @elapsed accelerated = fit(model, X, y; context)
    accelerated_fit_steady = median([
        @elapsed fit(model, X, y; context) for _ in 1:fit_samples])
    accelerator_operation = () -> predict_proba(accelerated, X)
    accelerator_first = @elapsed accelerator_operation()
    accelerator_steady = median([@elapsed accelerator_operation() for _ in 1:3])
    diagnostics = report(accelerated).details

    println((benchmark=:accelerator_scaling, observations,
        features=size(X, 2), device=diagnostics.device,
        fit_wall_seconds=accelerated_fit,
        warm_fit_wall_seconds=accelerated_fit_steady,
        cpu_warm_fit_wall_seconds=cpu_fit_steady,
        warm_fit_speedup=cpu_fit_steady / accelerated_fit_steady,
        compilation_seconds=diagnostics.compilation_nanoseconds / 1e9,
        first_call_seconds=accelerator_first,
        steady_state_seconds=accelerator_steady,
        cpu_first_call_seconds=cpu_first,
        cpu_steady_state_seconds=cpu_steady,
        cpu_fit_wall_seconds=cpu_fit,
        transferred_bytes=diagnostics.transferred_bytes,
        cache_hits=diagnostics.compilation_cache_hits,
        compilation_count=diagnostics.compilation_count,
        cache_size=diagnostics.compilation_cache_size,
        cache_capacity=diagnostics.compilation_cache_capacity,
        cache_evictions=diagnostics.compilation_cache_evictions,
        static_signature=(batch=true, features=true, element_type=true)))
end
