@testset "Multinomial naive Bayes" begin
    X = Float64[3 0 1; 2 0 0; 0 3 1; 0 2 2; 4 0 0; 0 4 1]
    y = [:left, :left, :right, :right, :left, :right]
    fitted = fit(MultinomialNaiveBayes(alpha=1.0), X, y)
    probabilities = predict_proba(fitted, X)

    @test predict(fitted, X) == y
    @test vec(sum(probabilities; dims=2)) ≈ ones(size(X, 1))
    @test fitted.classes == [:left, :right]
    feature_probabilities = exp.(fitted.feature_log_probabilities)
    @test all(isapprox(sum(row), 1.0) for row in eachrow(feature_probabilities))
    @test predict(fit(MultinomialNaiveBayes(), sparse(X), y), sparse(X)) == y
    uniform_prior = fit(MultinomialNaiveBayes(fit_prior=false), X, y)
    @test size(predict_proba(uniform_prior, X)) == (6, 2)
    @test_throws Tilia.UnsupportedDataError fit(MultinomialNaiveBayes(), X .- 1, y)
    @test_throws Tilia.SchemaMismatchError fit(MultinomialNaiveBayes(), X, y[1:end-1])
    @test_throws Tilia.InvalidHyperparameterError MultinomialNaiveBayes(alpha=0)
end
