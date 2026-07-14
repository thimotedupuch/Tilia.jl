using Tilia
using Tilia.Kernels
using Random

rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 32)
    elapsed = @elapsed pairwise_distances(view(X, 1:min(observations, 1_000), :), view(X, 1:100, :))
    println((kernel=:pairwise_euclidean, observations=observations,
             features=32, seconds=elapsed))
end
