using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 32)
    query = randn(rng, min(observations, 500), 32)
    for threaded in (false, true)
        operation = () -> Tilia.Kernels.pairwise_distance_blocks(
            query, X; block_size=100, threaded)
        first_call = @elapsed operation()
        elapsed = median([@elapsed operation() for _ in 1:3])
        println((benchmark=:cpu_scaling, observations,
                 threads=Threads.nthreads(), threaded,
                 first_call_seconds=first_call,
                 steady_state_seconds=elapsed))
    end
end
end
