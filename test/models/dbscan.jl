@testset "DBSCAN density clustering" begin
    X = Float64[0 0; 0.1 0; 0 0.1; 3 3; 3.1 3; 3 3.1; 8 8]
    fitted = fit(DBSCAN(radius=0.25, min_neighbors=3), X)
    @test fitted.labels == [1, 1, 1, 2, 2, 2, 0]
    @test fitted.core_indices == collect(1:6)
    @test report(fitted).details.clusters == 2
    @test report(fitted).details.noise_observations == 1
    @test predict(fitted, [0.05 0.05; 3.05 3.05; 10.0 10.0]) == [1, 2, 0]
    @test predict(fit(DBSCAN(radius=0.01, min_neighbors=2), X), X) == zeros(Int, 7)
    @test_throws Tilia.InvalidHyperparameterError DBSCAN(radius=0)
    @test_throws Tilia.InvalidHyperparameterError DBSCAN(min_neighbors=0)
    @test_throws Tilia.SchemaMismatchError predict(fitted, ones(2, 3))
end
