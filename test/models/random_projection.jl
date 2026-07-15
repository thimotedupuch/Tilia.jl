using Random: Xoshiro

@testset "Random projection" begin
    rng = Xoshiro(4)
    X = randn(rng, Float64, 40, 60)
    gaussian = fit(RandomProjection(n_components=30), X;
                   context=FitContext(seed=91))
    repeated = fit(RandomProjection(n_components=30), X;
                   context=FitContext(seed=91))
    projected = transform(gaussian, X)
    @test gaussian.projection == repeated.projection
    @test size(projected) == (40, 30)
    original_distances = Tilia.Kernels.pairwise_distances(X)
    projected_distances = Tilia.Kernels.pairwise_distances(projected)
    mask = triu!(trues(size(X, 1), size(X, 1)), 1)
    ratios = projected_distances[mask] ./ original_distances[mask]
    @test 0.8 < median(ratios) < 1.2

    sparse_model = fit(RandomProjection(n_components=20, distribution=:sparse), X;
                       context=FitContext(seed=92))
    @test count(iszero, sparse_model.projection) > length(sparse_model.projection) ÷ 2
    @test transform(sparse_model, sparse(X)) ≈ transform(sparse_model, X)
    @test eltype(transform(fit(RandomProjection(), Float32.(X)), Float32.(X))) == Float32
    @test_throws Tilia.UnsupportedDataError fit(RandomProjection(n_components=61), X)
    @test_throws Tilia.InvalidHyperparameterError RandomProjection(n_components=0)
    @test_throws Tilia.InvalidHyperparameterError RandomProjection(distribution=:bad)
    @test_throws Tilia.InvalidHyperparameterError RandomProjection(distribution=:sparse, density=0)
end
