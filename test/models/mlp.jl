@testset "Shallow multilayer perceptrons" begin
    X = [-1.0 -1; -1 1; 1 -1; 1 1]
    y = [:same, :different, :different, :same]
    model = MLPClassifier(hidden_units=8, activation=:tanh, learning_rate=0.2,
                          max_iterations=5_000, tolerance=1e-9)
    fitted = fit(model, X, y)
    @test predict(fitted, X) == y
    probabilities = predict_proba(fitted, X)
    @test vec(sum(probabilities; dims=2)) ≈ ones(4)
    @test last(report(fitted).details.objective_history) <
          first(report(fitted).details.objective_history)
    repeated = fit(model, X, y)
    @test repeated.input_weights == fitted.input_weights
    @test predict_proba(repeated, X) == probabilities

    Xr = reshape(collect(range(-1.0, 1.0; length=40)), :, 1)
    yr = @. 2 * Xr[:, 1] - 0.5
    regressor = fit(MLPRegressor(hidden_units=8, activation=:tanh,
        learning_rate=0.05, max_iterations=3_000, tolerance=1e-9), Xr, yr)
    @test root_mean_squared_error(yr, predict(regressor, Xr)) < 0.05
    @test report(regressor).details.optimizer == :batch_gradient_descent

    weights = ones(40)
    weights[end] = 2
    weighted = fit(MLPRegressor(hidden_units=4, max_iterations=2), Xr, yr; weights=weights)
    @test all(isfinite, predict(weighted, Xr))

    X32 = Float32.(X)
    fitted32 = fit(MLPClassifier(hidden_units=4, max_iterations=2), X32, y)
    @test eltype(fitted32.input_weights) == Float32
    @test eltype(predict_proba(fitted32, X32)) == Float32
    @test capabilities(MLPClassifier()).probabilistic
    @test_throws Tilia.InvalidHyperparameterError MLPClassifier(hidden_units=0)
    @test_throws Tilia.InvalidHyperparameterError MLPRegressor(activation=:bad)
end
