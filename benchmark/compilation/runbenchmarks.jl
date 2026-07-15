using Statistics
using Tilia

# Measure method compilation separately from first execution in this process.
X = reshape(collect(Float64, 1:1_600), 100, 16)
y = vec(sum(X; dims=2))
fit_operation = () -> fit(RidgeRegression(lambda=1.0), X, y)
fit_compilation_and_first = @elapsed fitted = fit_operation()
fit_steady = median([@elapsed fit_operation() for _ in 1:3])

query = view(X, 1:20, :)
inference_operation = () -> predict(fitted, query)
inference_compilation_and_first = @elapsed inference_operation()
inference_steady = median([@elapsed inference_operation() for _ in 1:3])

println((benchmark=:compilation_latency, model=:ridge,
    fit_compilation_and_first_seconds=fit_compilation_and_first,
    fit_steady_state_seconds=fit_steady,
    fit_estimated_compilation_seconds=max(0.0, fit_compilation_and_first - fit_steady),
    inference_compilation_and_first_seconds=inference_compilation_and_first,
    inference_steady_state_seconds=inference_steady,
    inference_estimated_compilation_seconds=max(0.0,
        inference_compilation_and_first - inference_steady)))
