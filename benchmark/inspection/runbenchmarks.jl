using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, Float32, observations, 16)
    y = 3 .* X[:, 1] .- X[:, 2] .+ 0.05f0 .* randn(rng, Float32, observations)
    fitted = fit(RidgeRegression(lambda=0.1f0), X, y)
    operation = () -> permutation_importance(
        fitted, X, y; n_repeats=3, context=FitContext(seed=42))
    first = @elapsed operation()
    steady = median([@elapsed operation() for _ in 1:3])
    println((benchmark=:permutation_importance, observations,
             features=size(X, 2), repeats=3,
             first_call_seconds=first, steady_state_seconds=steady))
end
end
