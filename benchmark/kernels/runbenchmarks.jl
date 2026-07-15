using Tilia
using Tilia.Kernels
using Random
using Statistics

let rng = Xoshiro(42)
function measure_kernel(operation, kernel, observations)
    first_call = @elapsed operation()
    steady = median([@elapsed operation() for _ in 1:3])
    println((benchmark=:kernel, kernel, observations,
             first_call_seconds=first_call, steady_state_seconds=steady,
             bytes=@allocated(operation())))
end

for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 32)
    vector = vec(view(X, :, 1))
    measure_kernel(() -> pairwise_distances(
        view(X, 1:min(observations, 500), :), X),
        :pairwise_euclidean, observations)
    measure_kernel(() -> reduction_sum(vector; stable=true),
                   :stable_reduction, observations)
    measure_kernel(() -> softmax(X; dims=2), :softmax, observations)
    measure_kernel(() -> covariance_matrix(X), :covariance, observations)
    measure_kernel(() -> topk_indices(vector, min(10, observations)),
                   :topk, observations)
end
end
