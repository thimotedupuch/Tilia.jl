@testset "Multinomial naive Bayes scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "multinomial_naive_bayes_sklearn.toml"))
    case = fixture["case"]
    X = reduce(vcat, permutedims.(case["X"]))
    fitted = fit(MultinomialNaiveBayes(), X, case["y"])
    atol = fixture["tolerance"]["absolute"]
    rtol = fixture["tolerance"]["relative"]
    expected_logs = reduce(vcat, permutedims.(case["feature_log_probabilities"]))
    expected_probabilities = reduce(vcat, permutedims.(case["probabilities"]))
    @test fitted.classes == case["classes"]
    @test fitted.feature_log_probabilities ≈ expected_logs atol=atol rtol=rtol
    @test predict_proba(fitted, X) ≈ expected_probabilities atol=atol rtol=rtol
end
