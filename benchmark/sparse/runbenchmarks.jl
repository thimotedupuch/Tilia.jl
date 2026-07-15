using Random
using SparseArrays
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    X = sprand(rng, Float64, observations, 100, 0.05)
    y = randn(rng, observations)
    vector = randn(rng, 100)

    matvec = () -> Tilia.Kernels.sparse_matvec(X, vector)
    matvec_first = @elapsed matvec()
    println((benchmark=:sparse_matvec, observations,
             first_call_seconds=matvec_first,
             steady_state_seconds=median([@elapsed matvec() for _ in 1:3]),
             bytes=@allocated(matvec())))

    operation = () -> fit(Lasso(lambda=0.05, max_iterations=20), X, y)
    first_call = @elapsed operation()
    elapsed = median([@elapsed operation() for _ in 1:3])
    println((benchmark=:sparse_lasso_fit, observations,
             features=size(X, 2), first_call_seconds=first_call,
             steady_state_seconds=elapsed))
end
end
