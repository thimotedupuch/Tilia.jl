using Tilia
using Random
using Statistics

let rng = Xoshiro(42)

function timed_training(operation, label)
    first_call = @elapsed result = operation()
    steady_state = median([@elapsed operation() for _ in 1:3])
    println((benchmark=:fit_time, label...,
             first_call_seconds=first_call,
             steady_state_seconds=steady_state))
    result
end

for observations in (100, 1_000, 10_000)
    features = 16
    X = randn(rng, observations, features)
    y = X * randn(rng, features)
    timed_training(() -> fit(RidgeRegression(lambda=1.0), X, y),
                   (model=:ridge, observations, features))
    timed_training(() -> fit(PCA(n_components=8), X),
                   (model=:pca, observations, features))
end
end
