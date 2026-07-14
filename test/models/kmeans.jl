@testset "KMeans clustering" begin
    X = [-0.1 0.0; 0.0 0.1; 0.1 -0.1; 9.9 10.0; 10.0 10.1; 10.1 9.9]
    model = KMeans(n_clusters=2, n_init=4, max_iterations=100)
    first_fit = fit(model, X)
    second_fit = fit(model, X)
    @test first_fit.centers == second_fit.centers
    @test first_fit.labels == second_fit.labels
    @test first_fit.converged
    @test first_fit.inertia < 0.2
    @test length(unique(first_fit.labels[1:3])) == 1
    @test length(unique(first_fit.labels[4:6])) == 1
    @test first_fit.labels[1] != first_fit.labels[4]
    @test predict(first_fit, X) == first_fit.labels
    distances = transform(first_fit, X)
    @test size(distances) == (6, 2)
    @test all(distances .>= 0)
    @test report(first_fit).details.n_clusters == 2
    @test all(diff(report(first_fit).details.objective_history) .<= 10eps())

    X32 = Float32.(X)
    fitted32 = fit(KMeans(n_clusters=2, n_init=1), X32)
    @test eltype(fitted32.centers) == Float32
    @test eltype(transform(fitted32, X32)) == Float32

    @test_throws Tilia.InvalidHyperparameterError KMeans(n_clusters=0)
    @test_throws Tilia.InvalidHyperparameterError KMeans(init=:bad)
    @test_throws Tilia.UnsupportedDataError fit(KMeans(n_clusters=3), ones(2, 2))
    @test_throws Tilia.SchemaMismatchError predict(first_fit, ones(2, 3))
end
