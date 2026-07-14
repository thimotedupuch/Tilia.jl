@testset "Semantic graph Chain" begin
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0]
    y = [2.0, 4.0, 9.0]
    chain = Chain(Standardize(), MeanRegressor())
    fitted = fit(chain, X, y)

    @test predict(fitted, X[1:2, :]) == fill(5.0, 2)
    @test report(fitted).details.execution == :semantic_graph
    @test report(fitted).details.nodes == 2
    standardized = transform(fitted.fitted_nodes[1], X)
    @test vec(mean(standardized; dims=1)) ≈ zeros(2) atol=1e-12
    @test vec(std(standardized; dims=1, corrected=false)) ≈ ones(2) atol=1e-12
    @test transform(fitted.fitted_nodes[1], inverse_transform(fitted.fitted_nodes[1], standardized)) ≈ standardized

    @test_throws Tilia.GraphValidationError fit(Chain(MeanRegressor(), Standardize()), X, y)
    @test_throws Tilia.InvalidHyperparameterError Chain()
    @test_throws Tilia.UnsupportedDataError fit(Standardize(), sparse(X))
    sparse_fitted = fit(Standardize(center=false), sparse(X))
    @test transform(sparse_fitted, sparse(X)) ≈ transform(sparse_fitted, X)
end

@testset "Weighted Chain fitting" begin
    X = [0.0 0; 1 1; 10 10]
    y = [0.0, 1.0, 100.0]
    weights = [10.0, 10.0, 0.1]
    pipeline = Chain(Standardize(), MeanRegressor())
    fitted = fit(pipeline, X, y; weights=weights)
    @test predict(fitted, X) == fill(sum(weights .* y) / sum(weights), 3)
    @test report(fitted).details.weighted
    @test report(last(fitted.fitted_nodes)).details.weighted
end
