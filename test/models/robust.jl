using Test
using Statistics
using Tilia

@testset "Robust Regression Models (Huber, Quantile, RANSAC, Theil-Sen)" begin
    X = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0; 5.0 4.0; 6.0 2.0]
    y = [2.0, 1.0, 3.0, 7.0, 8.0, 6.0]
    weights = [1.0, 1.2, 1.0, 1.5, 1.0, 1.0]

    # 1. Huber Regression
    huber = fit(HuberRegression(), X, y)
    preds_h = predict(huber, X)
    @test length(preds_h) == 6
    @test capabilities(HuberRegression) == (
        task=:regression, sparse=false, missing=false, weights=true,
        partial_fit=false, probabilistic=false,
    )

    # 2. Quantile Regression
    quantile_reg = fit(QuantileRegression(quantile=0.5), X, y)
    preds_q = predict(quantile_reg, X)
    @test length(preds_q) == 6
    @test capabilities(QuantileRegression) == (
        task=:regression, sparse=false, missing=false, weights=true,
        partial_fit=false, probabilistic=false,
    )

    # 3. RANSAC Regression
    ransac = fit(RANSACRegression(max_trials=50), X, y)
    preds_r = predict(ransac, X)
    @test length(preds_r) == 6
    @test sum(ransac.inlier_mask) >= 3
    @test ransac.report.details.trials_attempted <= 50
    @test ransac.report.details.stop_probability == 0.99
    @test capabilities(RANSACRegression) == (
        task=:regression, sparse=false, missing=false, weights=true,
        partial_fit=false, probabilistic=false,
    )

    exact_X = reshape(collect(1.0:8.0), :, 1)
    exact_y = 2 .* vec(exact_X) .+ 1
    early = fit(RANSACRegression(min_samples=2, residual_threshold=1e-10,
                                max_trials=100, stop_probability=0.99),
                exact_X, exact_y; context=FitContext(seed=12))
    @test early.report.details.trials_attempted == 1
    @test early.report.details.trial_limit == 1
    @test !capabilities(RANSACRegression(
        base_estimator=KNeighborsRegressor())).weights

    # 4. Theil-Sen Regression
    theilsen = fit(TheilSenRegression(max_subpopulations=100), X, y)
    preds_ts = predict(theilsen, X)
    @test length(preds_ts) == 6
    @test capabilities(TheilSenRegression) == (
        task=:regression, sparse=false, missing=false, weights=false,
        partial_fit=false, probabilistic=false,
    )
    @test_throws Tilia.UnsupportedDataError fit(
        TheilSenRegression(), X, y; weights=weights)

    # Check invalid parameters
    @test_throws Tilia.InvalidHyperparameterError HuberRegression(epsilon=0.5)
    @test_throws Tilia.InvalidHyperparameterError QuantileRegression(quantile=1.5)
    @test_throws Tilia.InvalidHyperparameterError RANSACRegression(max_trials=-5)
    @test_throws Tilia.InvalidHyperparameterError TheilSenRegression(max_subpopulations=0)
end
