using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 16)
    y = X * randn(rng, 16)
    for (name, fitted) in (
        (:ridge, fit(RidgeRegression(), X, y)),
        (:pca, fit(PCA(n_components=8), X)),
    )
        operation = name === :pca ? () -> transform(fitted, X) : () -> predict(fitted, X)
        first_call = @elapsed operation()
        steady_state = median([@elapsed operation() for _ in 1:3])
        println((benchmark=:memory, model=name, observations,
                 first_call_seconds=first_call, steady_state_seconds=steady_state,
                 inference_bytes=@allocated(operation())))
    end
end
end
