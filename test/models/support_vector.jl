@testset "Kernel support-vector models" begin
    X = [-1.0 -1; -1 1; 1 -1; 1 1]
    y = [:same, :different, :different, :same]
    classifier = fit(SupportVectorClassifier(C=100.0, kernel=:rbf, gamma=1.0,
        max_iterations=10_000, tolerance=1e-8), X, y)
    @test predict(classifier, X) == y
    @test report(classifier).details.loss == :squared_hinge
    @test all(count -> count > 0, report(classifier).details.support_vectors)
    @test all(diff(first(report(classifier).details.objective_history)) .<= 1e-8)

    Xr = reshape(collect(range(-2.0, 2.0; length=20)), :, 1)
    yr = @. 1.5 * Xr[:, 1] - 0.25
    regressor = fit(SupportVectorRegressor(C=100.0, epsilon=0.01,
        kernel=:linear, max_iterations=10_000, tolerance=1e-8), Xr, yr)
    @test root_mean_squared_error(yr, predict(regressor, Xr)) < 0.1
    @test report(regressor).details.loss == :squared_epsilon_insensitive
    @test all(diff(report(regressor).details.objective_history) .<= 1e-8)

    weights = ones(20)
    weights[end] = 2
    weighted = fit(SupportVectorRegressor(C=20.0, epsilon=0.05,
        kernel=:linear), Xr, yr; weights=weights)
    @test all(isfinite, predict(weighted, Xr))

    X32 = Float32.(X)
    fitted32 = fit(SupportVectorClassifier(C=20.0f0, kernel=:rbf,
        max_iterations=100), X32, y)
    @test eltype(fitted32.coefficients) == Float32
    @test capabilities(SupportVectorClassifier()).task == :classification
    @test_throws Tilia.InvalidHyperparameterError SupportVectorClassifier(C=0)
    @test_throws Tilia.InvalidHyperparameterError SupportVectorRegressor(epsilon=-1)
end
