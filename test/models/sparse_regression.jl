@testset "Lasso and elastic-net regression" begin
    X = [1.0 0 1; 0 1 1; 1 1 1; 2 1 1; 1 2 1; 3 2 1]
    y = 3 .* X[:, 1] .- 2 .* X[:, 2] .+ 0.5
    for model in (Lasso(lambda=0.0, tolerance=1e-10, max_iterations=10_000),
                  ElasticNet(lambda=0.0, tolerance=1e-10, max_iterations=10_000))
        fitted = fit(model, X, y)
        @test predict(fitted, X) ≈ y atol=1e-6
        @test fitted.coefficients[1:2] ≈ [3.0, -2.0] atol=1e-5
        @test report(fitted).details.converged
        @test all(diff(report(fitted).details.objective_history) .<= 100eps())
    end

    sparse_model = Lasso(lambda=0.2, tolerance=1e-9, max_iterations=10_000)
    dense_fit = fit(sparse_model, X, y)
    sparse_fit = fit(sparse_model, sparse(X), y)
    @test sparse_fit.coefficients ≈ dense_fit.coefficients atol=1e-8
    @test sparse_fit.intercept ≈ dense_fit.intercept atol=1e-8
    @test predict(sparse_fit, sparse(X)) ≈ predict(dense_fit, X)

    strongly_regularized = fit(Lasso(lambda=10.0), X, y)
    @test count(coefficient -> !iszero(coefficient), strongly_regularized.coefficients) == 0

    weights = [2.0, 1, 1, 1, 1, 2]
    weighted = fit(ElasticNet(lambda=0.1), X, y; weights=weights)
    repeated = [1, 1, 2, 3, 4, 5, 6, 6]
    replicated = fit(ElasticNet(lambda=0.1), X[repeated, :], y[repeated])
    @test weighted.coefficients ≈ replicated.coefficients atol=1e-6
    @test weighted.intercept ≈ replicated.intercept atol=1e-6

    X32, y32 = Float32.(X), Float32.(y)
    @test eltype(predict(fit(Lasso(lambda=0.1f0), X32, y32), X32)) == Float32
    @test capabilities(Lasso()).sparse
    @test_throws Tilia.InvalidHyperparameterError Lasso(lambda=-1)
    @test_throws Tilia.InvalidHyperparameterError ElasticNet(l1_ratio=1.1)
end
