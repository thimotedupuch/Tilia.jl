using Random: Xoshiro

@testset "Partial least-squares regression" begin
    rng = Xoshiro(33)
    latent = randn(rng, 120, 2)
    X = latent * [1.0 0.8 0.2 0.0; 0.1 0.4 1.0 0.7] .+
        0.01 .* randn(rng, 120, 4)
    y = 2 .* latent[:, 1] .- 1.2 .* latent[:, 2] .+
        0.01 .* randn(rng, 120)
    fitted = fit(PartialLeastSquaresRegression(n_components=2), X, y)
    @test root_mean_squared_error(y, predict(fitted, X)) < 0.05
    scores = transform(fitted, X)
    @test size(scores) == (120, 2)
    @test size(inverse_transform(fitted, scores)) == size(X)
    @test report(fitted).details.solver == :nipals
    @test report(fitted).details.components == 2

    fitted32 = fit(PartialLeastSquaresRegression(n_components=2),
                   Float32.(X), Float32.(y))
    @test eltype(predict(fitted32, Float32.(X))) == Float32
    @test eltype(transform(fitted32, Float32.(X))) == Float32
    @test_throws Tilia.UnsupportedDataError fit(
        PartialLeastSquaresRegression(n_components=5), X, y)
    @test_throws Tilia.InvalidHyperparameterError PartialLeastSquaresRegression(n_components=0)
    @test_throws Tilia.SchemaMismatchError predict(fitted, ones(3, 5))
end
