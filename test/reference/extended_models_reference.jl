@testset "Extended supervised-model external references" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "extended_models_sklearn.toml"))
    @test fixture["source"]["scikit_learn_version"] == "1.9.0"
    @test fixture["source"]["scipy_version"] == "1.18.0"

    matrix(rows) = reduce(vcat, permutedims.(rows))
    input = fixture["input"]
    tolerance = fixture["tolerance"]
    X_regression = matrix(input["X_regression"])
    query_regression = matrix(input["query_regression"])
    y_positive = input["y_positive"]
    y_robust = input["y_robust"]

    for (model, key) in (
        (PoissonRegression(lambda=0.0, max_iterations=2000, tolerance=1e-9), "poisson"),
        (GammaRegression(lambda=0.0, max_iterations=2000, tolerance=1e-9), "gamma"),
        (TweedieRegression(power=1.5, lambda=0.0, max_iterations=2000,
                           tolerance=1e-9), "tweedie"),
    )
        actual = predict(fit(model, X_regression, y_positive), query_regression)
        @test actual ≈ fixture["glm"][key] atol=tolerance["glm_absolute"] rtol=tolerance["glm_relative"]
    end

    robust_models = (
        (QuantileRegression(quantile=0.5, lambda=0.0, epsilon=1e-6,
                            max_iterations=3000, tolerance=1e-9), "quantile"),
        (HuberRegression(epsilon=1.35, lambda=0.0, max_iterations=3000,
                         tolerance=1e-9), "huber"),
        (TheilSenRegression(max_subpopulations=10000), "theil_sen"),
        (RANSACRegression(min_samples=3, residual_threshold=0.5, max_trials=100,
                          stop_probability=0.99), "ransac"),
    )
    for (model, key) in robust_models
        actual = predict(fit(model, X_regression, y_robust;
                             context=FitContext(seed=0)), query_regression)
        @test actual ≈ fixture["robust"][key] atol=tolerance["robust_absolute"] rtol=tolerance["robust_relative"]
    end

    X_ordinal = matrix(input["X_ordinal"])
    query_ordinal = matrix(input["query_ordinal"])
    ordinal = fit(OrdinalRegression(lambda=0.0, max_iterations=2000,
                                    tolerance=1e-9),
                  X_ordinal, input["y_ordinal"])
    @test fixture["ordinal"]["optimizer_success"]
    @test predict_proba(ordinal, query_ordinal) ≈ matrix(fixture["ordinal"]["probabilities"]) atol=tolerance["ordinal_absolute"] rtol=tolerance["ordinal_relative"]

    X_binary = matrix(input["X_binary"])
    query_binary = matrix(input["query_binary"])
    calibrated = fit(CalibratedClassifier(
        LogisticRegression(lambda=0.0, max_iterations=2000, tolerance=1e-10);
        cv=KFold(3)), X_binary, input["y_binary"])
    @test calibrated.Platt_A[2] ≈ fixture["calibration"]["platt_parameters"][1] atol=tolerance["calibration_absolute"] rtol=tolerance["calibration_relative"]
    @test calibrated.Platt_B[2] ≈ fixture["calibration"]["platt_parameters"][2] atol=tolerance["calibration_absolute"] rtol=tolerance["calibration_relative"]
    @test predict_proba(calibrated, query_binary) ≈ matrix(fixture["calibration"]["probabilities"]) atol=tolerance["calibration_absolute"] rtol=tolerance["calibration_relative"]

    stacking = StackingRegressor(
        (LinearRegression(), RidgeRegression(lambda=0.5)),
        RidgeRegression(lambda=0.25);
        cv=KFold(3),
    )
    actual_stacking = predict(fit(stacking, X_regression, y_robust), query_regression)
    @test actual_stacking ≈ fixture["stacking_regression"]["prediction"] atol=tolerance["stacking_absolute"] rtol=tolerance["stacking_relative"]
end
