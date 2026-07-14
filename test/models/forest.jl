@testset "Random forests and extra trees" begin
    X = [0.0 0; 0 1; 1 0; 1 1; 2 0; 2 1; 3 0; 3 1]
    y_class = [:low, :low, :low, :low, :high, :high, :high, :high]
    for model in (RandomForestClassifier(n_estimators=25, max_features=nothing),
                  ExtraTreesClassifier(n_estimators=25, max_features=nothing))
        fitted = fit(model, X, y_class)
        probabilities = predict_proba(fitted, X)
        @test predict(fitted, X) == y_class
        @test vec(sum(probabilities; dims=2)) ≈ ones(8)
        @test length(fitted.trees) == 25
        @test sum(fitted.feature_importances) ≈ 1
        @test report(fitted).details.n_estimators == 25
        repeated = fit(model, X, y_class)
        @test predict_proba(repeated, X) == probabilities
    end

    y_regression = 2 .* X[:, 1] .- X[:, 2]
    for model in (RandomForestRegressor(n_estimators=30, max_features=nothing),
                  ExtraTreesRegressor(n_estimators=30, max_features=nothing))
        fitted = fit(model, X, y_regression)
        @test root_mean_squared_error(y_regression, predict(fitted, X)) < 1.0
        @test length(fitted.trees) == 30
        @test report(fitted).details.mean_nodes >= 1
    end

    weights = [1.0, 1, 1, 1, 2, 2, 2, 2]
    weighted = fit(RandomForestClassifier(n_estimators=5, bootstrap=false,
        max_features=nothing), X, y_class; weights=weights)
    @test predict(weighted, X) == y_class

    X32 = Float32.(X)
    fitted32 = fit(ExtraTreesRegressor(n_estimators=3), X32, Float32.(y_regression))
    @test eltype(predict(fitted32, X32)) == Float32
    @test capabilities(RandomForestClassifier()).probabilistic
    @test_throws Tilia.InvalidHyperparameterError RandomForestClassifier(n_estimators=0)
end
