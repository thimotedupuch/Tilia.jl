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
        @test isfile(joinpath(path, "manifest.toml"))
    end
end
