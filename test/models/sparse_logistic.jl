@testset "Sparse logistic regression" begin
    X = [-2.0 -1 0; -1 -2 0; -2 -2 1; 1 2 0; 2 1 0; 2 2 1]
    y = [:negative, :negative, :negative, :positive, :positive, :positive]
    model = SparseLogisticRegression(lambda=0.01, l1_ratio=0.8,
                                     tolerance=1e-8, max_iterations=10_000)
    fitted = fit(model, X, y)
    @test predict(fitted, X) == y
    probabilities = predict_proba(fitted, X)
    @test size(probabilities) == (6, 2)
    @test vec(sum(probabilities; dims=2)) ≈ ones(6)
    @test report(fitted).details.solver == :proximal_gradient
    @test all(report(fitted).details.convergence)
    @test all(diff(first(report(fitted).details.objective_history)) .<= 100eps())

    sparse_fit = fit(model, sparse(X), y)
    @test sparse_fit.coefficients ≈ fitted.coefficients atol=1e-8
    @test sparse_fit.intercept ≈ fitted.intercept atol=1e-8
    @test predict_proba(sparse_fit, sparse(X)) ≈ probabilities atol=1e-8

    strong = fit(SparseLogisticRegression(lambda=100.0), X, y)
    @test count(coefficient -> !iszero(coefficient), strong.coefficients) == 0

    weights = [2.0, 1, 1, 1, 1, 2]
    weighted = fit(model, X, y; weights=weights)
    repeated = [1, 1, 2, 3, 4, 5, 6, 6]
    replicated = fit(model, X[repeated, :], y[repeated])
    @test weighted.coefficients ≈ replicated.coefficients atol=1e-6
    @test weighted.intercept ≈ replicated.intercept atol=1e-6

    X32 = Float32.(X)
    fitted32 = fit(SparseLogisticRegression(lambda=0.01f0, tolerance=1f-6), X32, y)
    @test eltype(predict_proba(fitted32, X32)) == Float32
    @test capabilities(SparseLogisticRegression()).sparse
    @test_throws Tilia.InvalidHyperparameterError SparseLogisticRegression(lambda=-1)
    @test_throws Tilia.InvalidHyperparameterError SparseLogisticRegression(l1_ratio=-0.1)
end
