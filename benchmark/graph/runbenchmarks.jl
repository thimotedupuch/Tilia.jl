using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = randn(rng, observations, 16)
    y = X * randn(rng, 16)
    pipeline = fit(Chain(Standardize(center=true, scale=false),
                         Standardize(center=false, scale=true),
                         RidgeRegression(lambda=1.0)), X, y)
    optimized = Tilia.optimize(pipeline)
    for (name, fitted) in (:unoptimized => pipeline, :optimized => optimized)
        first_call = @elapsed predict(fitted, X)
        elapsed = median([@elapsed predict(fitted, X) for _ in 1:3])
        println((benchmark=:graph_optimization, graph=name, observations,
                 nodes=length(fitted.fitted_nodes), first_call_seconds=first_call,
                 steady_state_seconds=elapsed,
                 bytes=@allocated(predict(fitted, X))))
    end
end
end
