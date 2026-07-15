@testset "MeanRegressor" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y = [1.0, 2.0, 6.0]
    model = MeanRegressor()
    fitted = fit(model, X, y)

    @test predict(fitted, X) == fill(3.0, 3)
    @test report(fitted).status == :success
    @test report(fitted).observations == 3
    @test fitted.model === model
    @test fit(model, X, y; weights=[1.0, 1.0, 2.0]).mean == 3.75
    @test predict(fit(model, sparse(X), y), sparse(X)) == fill(3.0, 3)
    online = partial_fit(partial_fit(model, X[1:2, :], y[1:2]), X[3:3, :], y[3:3])
    @test online.mean ≈ fitted.mean
    @test report(online).observations == 3
    @test report(online).details.partial_updates == 1
    @test capabilities(MeanRegressor()).partial_fit

    @test_throws Tilia.SchemaMismatchError fit(model, X, y[1:2])
    @test_throws Tilia.SchemaMismatchError predict(fitted, ones(2, 3))
    @test_throws Tilia.UnsupportedDataError fit(model, X, [1.0, NaN, 2.0])
    @test_throws Tilia.UnsupportedDataError fit(model, X, y; weights=zeros(3))
    @test_throws Tilia.UnsupportedBackendError fit(model, X, y;
        context=FitContext(backend=ReactantBackend()))
    @test_throws Tilia.InvalidHyperparameterError ReactantBackend(device=:quantum)

    mktempdir() do directory
        path = joinpath(directory, "model")
        @test save_model(path, fitted) == path
        loaded = load_model(path)
        @test predict(loaded, X) == predict(fitted, X)
        @test loaded.model isa MeanRegressor
        @test report(loaded).root_seed == report(fitted).root_seed
        @test report(loaded).stream_id == report(fitted).stream_id
        @test isfile(joinpath(path, "manifest.toml"))
    end
end

@testset "Deterministic context substreams" begin
    root = FitContext(seed=42)
    first = derive_context(root, :graph_node, 1)
    repeated = derive_context(root, :graph_node, 1)
    second = derive_context(root, :graph_node, 2)
    @test first.root_seed == repeated.root_seed == UInt64(42)
    @test first.stream_id == repeated.stream_id == "root/graph_node/1"
    @test rand(first.rng, UInt64, 4) == rand(repeated.rng, UInt64, 4)
    @test rand(derive_context(root, :graph_node, 1).rng, UInt64, 4) !=
          rand(second.rng, UInt64, 4)

    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y = [1.0, 2.0, 6.0]
    fitted = fit(Chain(Standardize(), MeanRegressor()), X, y; context=root)
    @test report(fitted).root_seed == UInt64(42)
    @test report(fitted).stream_id == "root"
    @test report(fitted).deterministic
    @test report(fitted).thread_count == Threads.nthreads()
end
