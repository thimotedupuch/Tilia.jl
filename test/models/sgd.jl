@testset "Streaming SGD linear models" begin
    X = hcat([sin(i / 3) for i in 1:80],
             [cos(i / 5) for i in 1:80],
             [i / 20 for i in 1:80])
    y = 2 .* X[:, 1] .- 0.7 .* X[:, 2] .+ 0.3 .* X[:, 3] .+ 0.2
    regressor = SGDRegressor(learning_rate=0.2, l2=0.0, epochs=100,
                             batch_size=16)
    fitted_regressor = fit(regressor, X, y; context=FitContext(seed=41))
    @test root_mean_squared_error(y, predict(fitted_regressor, X)) < 0.08
    @test eltype(predict(fit(regressor, Float32.(X), Float32.(y)), Float32.(X))) == Float32
    @test predict(fit(regressor, sparse(X), y; context=FitContext(seed=41)), sparse(X)) ≈
          predict(fitted_regressor, X) atol=1e-10

    online_regressor = partial_fit(regressor, X[1:40, :], y[1:40];
                                   context=FitContext(seed=7))
    first_updates = online_regressor.updates
    online_regressor = partial_fit(online_regressor, X[41:80, :], y[41:80];
                                   context=FitContext(seed=8))
    @test online_regressor.updates > first_updates
    @test report(online_regressor).observations == 80

    labels = ifelse.(y .> median(y), :high, :low)
    classifier = SGDClassifier(learning_rate=0.2, l2=1e-4, epochs=80,
                               batch_size=16)
    fitted_classifier = fit(classifier, X, labels; context=FitContext(seed=42))
    probabilities = predict_proba(fitted_classifier, X)
    @test accuracy_score(labels, predict(fitted_classifier, X)) >= 0.925
    @test vec(sum(probabilities; dims=2)) ≈ ones(80)
    @test fitted_classifier.classes == [:high, :low]

    online_classifier = partial_fit(classifier, X[1:20, :], labels[1:20];
        classes=[:high, :low], context=FitContext(seed=10))
    online_classifier = partial_fit(online_classifier, X[21:40, :], labels[21:40];
        context=FitContext(seed=11))
    @test online_classifier.classes == [:high, :low]
    @test size(predict_proba(online_classifier, X)) == (80, 2)
    @test_throws Tilia.SchemaMismatchError partial_fit(
        online_classifier, X[1:2, :], [:unknown, :unknown])
    @test_throws Tilia.InvalidHyperparameterError SGDRegressor(learning_rate=0)
    @test_throws Tilia.InvalidHyperparameterError SGDClassifier(batch_size=0)
end
