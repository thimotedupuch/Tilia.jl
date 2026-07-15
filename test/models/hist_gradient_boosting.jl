@testset "Histogram gradient boosting" begin
    X = reshape(collect(range(-3.0, 3.0; length=60)), :, 1)
    y_regression = @. X[:, 1]^2 + 0.5 * X[:, 1]
    regressor = fit(HistGradientBoostingRegressor(n_estimators=40,
        learning_rate=0.2, max_depth=2, max_bins=12, min_samples_leaf=2),
        X, y_regression)
    predictions = predict(regressor, X)
    baseline = fill(mean(y_regression), length(y_regression))
    @test root_mean_squared_error(y_regression, predictions) <
          0.4 * root_mean_squared_error(y_regression, baseline)
    @test maximum(length, regressor.bin_edges) <= 11
    @test all(diff(report(regressor).details.objective_history) .<= 1e-10)
    @test report(regressor).details.loss == :squared_error

    y_class = ifelse.(X[:, 1] .> 0, :positive, :negative)
    classifier = fit(HistGradientBoostingClassifier(n_estimators=40,
        learning_rate=0.3, max_depth=2, max_bins=10, min_samples_leaf=2), X, y_class)
    @test predict(classifier, X) == y_class
    probabilities = predict_proba(classifier, X)
    @test vec(sum(probabilities; dims=2)) ≈ ones(60)
    @test all(diff(report(classifier).details.objective_history) .<= 1e-10)
    @test report(classifier).details.bins_per_feature == [9]
    @test report(classifier).details.second_order

    regularized_model = HistGradientBoostingRegressor(n_estimators=12,
        max_bins=10, min_samples_leaf=2, l1_regularization=0.01,
        l2_regularization=0.1, row_subsample=0.8, feature_subsample=1.0)
    regularized_first = fit(regularized_model, X, y_regression;
                            context=FitContext(seed=77))
    regularized_second = fit(regularized_model, X, y_regression;
                             context=FitContext(seed=77))
    @test predict(regularized_first, X) == predict(regularized_second, X)
    @test report(regularized_first).details.row_subsample == 0.8
    suppressed = fit(HistGradientBoostingRegressor(n_estimators=2,
        max_bins=8, min_samples_leaf=2, l1_regularization=1e9), X, y_regression)
    @test predict(suppressed, X) ≈ fill(mean(y_regression), size(X, 1))

    weights = ones(60)
    weights[end-9:end] .= 2
    weighted = fit(HistGradientBoostingRegressor(n_estimators=5,
        max_bins=8, min_samples_leaf=2), X, y_regression; weights=weights)
    @test length(weighted.trees) == 5

    X32, y32 = Float32.(X), Float32.(y_regression)
    fitted32 = fit(HistGradientBoostingRegressor(n_estimators=3,
        max_bins=8, min_samples_leaf=2), X32, y32)
    @test eltype(predict(fitted32, X32)) == Float32
    @test capabilities(HistGradientBoostingClassifier()).probabilistic
    @test_throws Tilia.InvalidHyperparameterError HistGradientBoostingRegressor(max_bins=1)
    @test_throws Tilia.InvalidHyperparameterError HistGradientBoostingClassifier(learning_rate=0)
    @test_throws Tilia.InvalidHyperparameterError HistGradientBoostingRegressor(l1_regularization=-1)
    @test_throws Tilia.InvalidHyperparameterError HistGradientBoostingRegressor(row_subsample=0)
    @test_throws Tilia.InvalidHyperparameterError HistGradientBoostingClassifier(feature_subsample=1.1)
end
