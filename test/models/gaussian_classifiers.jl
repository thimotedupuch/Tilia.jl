@testset "Gaussian probabilistic classifiers" begin
    X = [-2.2 -1.8; -2.0 -2.1; -1.8 -2.2; 1.8 2.1; 2.0 1.9; 2.2 2.2]
    y = [:cold, :cold, :cold, :hot, :hot, :hot]
    for model in (GaussianNaiveBayes(), LinearDiscriminantAnalysis(),
                  QuadraticDiscriminantAnalysis())
        fitted = fit(model, X, y)
        probabilities = predict_proba(fitted, X)
        @test size(probabilities) == (6, 2)
        @test all(isfinite, probabilities)
        @test vec(sum(probabilities; dims=2)) ≈ ones(6)
        @test predict(fitted, X) == y
        @test fitted.classes == [:cold, :hot]
        @test report(fitted).details.class_order == [:cold, :hot]
        @test capabilities(model).probabilistic
    end

    weights = [2.0, 1, 1, 1, 1, 2]
    weighted = fit(GaussianNaiveBayes(), X, y; weights=weights)
    replicated_indices = [1, 1, 2, 3, 4, 5, 6, 6]
    replicated = fit(GaussianNaiveBayes(), X[replicated_indices, :], y[replicated_indices])
    @test weighted.means ≈ replicated.means
    @test weighted.variances ≈ replicated.variances
    @test weighted.priors ≈ replicated.priors

    X32 = Float32.(X)
    for model in (GaussianNaiveBayes(), LinearDiscriminantAnalysis(),
                  QuadraticDiscriminantAnalysis())
        @test eltype(predict_proba(fit(model, X32, y), X32)) == Float32
    end

    @test_throws Tilia.InvalidHyperparameterError GaussianNaiveBayes(var_smoothing=-1)
    @test_throws Tilia.InvalidHyperparameterError LinearDiscriminantAnalysis(regularization=-1)
    @test_throws Tilia.InvalidHyperparameterError QuadraticDiscriminantAnalysis(regularization=-1)
    @test_throws Tilia.SchemaMismatchError fit(GaussianNaiveBayes(), X, y[1:5])
    @test_throws Tilia.UnsupportedDataError fit(GaussianNaiveBayes(), X, fill(:one, 6))
end
