using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 16)
    y = X * randn(rng, 16)
    query = randn(rng, min(observations, 500), 16)
    for (name, fitted) in (
        :ridge => fit(RidgeRegression(lambda=1.0), X, y),
        :pca => fit(PCA(n_components=8), X),
        :neighbors => fit(NearestNeighbors(n_neighbors=5), X),
    )
        operation = name === :pca ? () -> transform(fitted, query) :
                    name === :neighbors ? () -> kneighbors(fitted, query) :
                    () -> predict(fitted, query)
        first_call = @elapsed operation()
        steady_state = median([@elapsed operation() for _ in 1:3])
        println((benchmark=:inference, model=name, observations,
                 first_call_seconds=first_call, steady_state_seconds=steady_state))
    end
end
end
