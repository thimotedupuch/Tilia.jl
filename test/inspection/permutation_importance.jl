@testset "Deterministic permutation importance" begin
    X = hcat([sin(index) for index in 1:80],
             [cos(3index) for index in 1:80])
    y = 4 .* X[:, 1] .+ 0.01 .* X[:, 2]
    fitted = fit(RidgeRegression(lambda=1e-6), X, y)
    original = copy(X)
    context = FitContext(seed=2026)
    first_result = permutation_importance(
        fitted, X, y; n_repeats=4, context)
    second_result = permutation_importance(
        fitted, X, y; n_repeats=4, context=FitContext(seed=2026))

    @test first_result.importances == second_result.importances
    @test first_result.mean_importance[1] > first_result.mean_importance[2]
    @test first_result.mean_importance[1] > 0
    @test first_result.feature_names == [:x1, :x2]
    @test size(first_result.importances) == (2, 4)
    @test all(first_result.standard_deviation .>= 0)
    @test X == original

    loss_result = permutation_importance(fitted, X, y;
        scoring=mean_squared_error, greater_is_better=false,
        n_repeats=3, context=FitContext(seed=17))
    @test loss_result.mean_importance[1] > 0

    labels = ifelse.(X[:, 1] .> 0, :positive, :negative)
    graph = fit(Chain(Standardize(), LogisticRegression(lambda=0.1)), X, labels)
    graph_result = permutation_importance(
        graph, X, labels; n_repeats=3, context=FitContext(seed=19))
    @test graph_result.baseline_score >= 0.9
    @test graph_result.mean_importance[1] > graph_result.mean_importance[2]

    table = (signal=X[:, 1], noise=X[:, 2])
    table_fitted = fit(RidgeRegression(lambda=1e-6), table, y)
    table_result = permutation_importance(
        table_fitted, table, y; n_repeats=2, context=FitContext(seed=23))
    @test table_result.feature_names == [:signal, :noise]
    dataset_result = permutation_importance(
        table_fitted, Dataset(table; target=y);
        n_repeats=2, context=FitContext(seed=23))
    @test dataset_result.importances == table_result.importances

    @test_throws ArgumentError permutation_importance(fitted, X, y; n_repeats=0)
    @test_throws DimensionMismatch permutation_importance(fitted, X, y[1:end-1])
    @test_throws ArgumentError permutation_importance(fitted, X, y;
        scoring=(truth, prediction) -> NaN)
    @test_throws ArgumentError permutation_importance(
        fitted, Dataset(X); n_repeats=2)
end
