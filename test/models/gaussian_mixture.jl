@testset "Gaussian mixture EM" begin
    X = [-3.2 -2.9; -3.0 -3.1; -2.8 -3.0; -3.1 -2.8;
          2.8 3.1; 3.0 2.9; 3.2 3.0; 3.1 3.2]
    model = GaussianMixture(n_components=2, n_init=3, max_iterations=100,
                            tolerance=1e-5, regularization=1e-5)
    fitted = fit(model, X)
    @test fitted.converged
    @test sum(fitted.mixture_weights) ≈ 1
    @test all(fitted.mixture_weights .> 0)
    ordered_means = fitted.means[sortperm(fitted.means[:, 1]), :]
    @test ordered_means[1, :] ≈ [-3.025, -2.95] atol=0.2
    @test ordered_means[2, :] ≈ [3.025, 3.05] atol=0.2
    responsibilities = predict_proba(fitted, X)
    @test size(responsibilities) == (8, 2)
    @test vec(sum(responsibilities; dims=2)) ≈ ones(8)
    labels = predict(fitted, X)
    @test length(unique(labels[1:4])) == 1
    @test length(unique(labels[5:8])) == 1
    @test labels[1] != labels[5]
    @test report(fitted).details.covariance_type == :full
    @test all(isfinite, report(fitted).details.objective_history)
    @test all(diff(report(fitted).details.objective_history) .>= -1e-8)

    repeated = fit(model, X)
    @test repeated.means == fitted.means
    @test repeated.mixture_weights == fitted.mixture_weights

    X32 = Float32.(X)
    fitted32 = fit(GaussianMixture(n_components=2, n_init=1), X32)
    @test eltype(fitted32.means) == Float32
    @test eltype(predict_proba(fitted32, X32)) == Float32

    @test_throws Tilia.InvalidHyperparameterError GaussianMixture(n_components=0)
    @test_throws Tilia.InvalidHyperparameterError GaussianMixture(regularization=0)
    @test_throws Tilia.UnsupportedDataError fit(GaussianMixture(n_components=3), ones(2, 2))
    @test_throws Tilia.SchemaMismatchError predict(fitted, ones(1, 3))
end
