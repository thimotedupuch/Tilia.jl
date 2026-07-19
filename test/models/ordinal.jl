using Test
using Statistics
using Tilia

@testset "Ordinal Regression Model" begin
    X = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0; 5.0 4.0; 6.0 2.0]
    y = [1, 1, 2, 2, 3, 3] # Ordinal classes: 1 < 2 < 3
    weights = [1.0, 1.0, 1.2, 1.0, 1.5, 1.0]

    # Fit Ordinal Regression
    ordinal = fit(OrdinalRegression(), X, y)

    # Check predictions
    preds = predict(ordinal, X)
    @test length(preds) == 6
    @test eltype(preds) == Int

    probs = predict_proba(ordinal, X)
    @test size(probs) == (6, 3)
    @test all(sum(probs; dims=2) ≈ ones(6))

    # Weighted fit
    ordinal_w = fit(OrdinalRegression(lambda=0.1), X, y; weights=weights)
    @test length(predict(ordinal_w, X)) == 6

    # Check capabilities
    @test capabilities(OrdinalRegression) == (
        task=:classification, sparse=false, missing=false, weights=true,
        partial_fit=false, probabilistic=true,
    )

    # Check invalid parameters
    @test_throws Tilia.InvalidHyperparameterError OrdinalRegression(lambda=-1.0)
    @test_throws Tilia.InvalidHyperparameterError OrdinalRegression(max_iterations=0)
end
