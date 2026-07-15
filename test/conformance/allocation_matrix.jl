@testset "Inference allocation budgets" begin
    for case in NUMERICAL_CONFORMANCE_CASES
        model = case.make(Float64)
        X, yreg, yclass = _conformance_data(Float64, model)
        fitted = _conformance_fit(model, X, yreg, yclass)
        _conformance_output(fitted, X) # compile before measurement
        allocation = @allocated _conformance_output(fitted, X)
        @test allocation <= 2_000_000
    end

    X = Float64[sin(i * j) for i in 1:64, j in 1:4]
    y = X[:, 1] .- 0.5 .* X[:, 2]
    pipeline = fit(Chain(Standardize(), PCA(n_components=2),
                         RidgeRegression(lambda=0.1)), X, y)
    predict(pipeline, X)
    graph_allocation = @allocated predict(pipeline, X)
    # Ordinary inference executes fitted nodes directly; numerical re-lowering
    # is reserved for explicit inspection and tracing.
    @test graph_allocation <= 20_000

    truth = repeat([:a, :b], 32)
    predicted = copy(truth)
    accuracy_score(truth, predicted)
    metric_allocation = @allocated accuracy_score(truth, predicted)
    @test metric_allocation <= 10_000

    destination = sparse(X)
    scales = ones(size(X, 2))
    Tilia.Kernels.scale_columns!(destination, scales)
    repeated_transform_allocation = @allocated begin
        Tilia.Kernels.scale_columns!(destination, scales)
        Tilia.Kernels.scale_columns!(destination, scales)
    end
    @test repeated_transform_allocation <= 256
end
