using Test
using Statistics
using Tilia

@testset "Generalized Linear Models (Poisson, Gamma, Tweedie)" begin
    X = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0]
    y = [2.0, 1.0, 3.0, 7.0]
    weights = [1.0, 1.5, 1.0, 2.0]

    # Poisson Regression
    poisson = fit(PoissonRegression(), X, y)
    preds = predict(poisson, X)
    @test length(preds) == 4
    @test all(p -> p > 0, preds)
    @test capabilities(PoissonRegression) == (
        task=:regression, sparse=false, missing=false, weights=true,
        partial_fit=false, probabilistic=false,
    )

    # Poisson with identity link
    poisson_id = fit(PoissonRegression(link=:identity), X, y)
    @test length(predict(poisson_id, X)) == 4

    # Gamma Regression (y must be strictly positive)
    gamma = fit(GammaRegression(), X, y)
    preds_g = predict(gamma, X)
    @test length(preds_g) == 4
    @test all(p -> p > 0, preds_g)

    # Tweedie Regression
    tweedie = fit(TweedieRegression(power=1.5), X, y)
    preds_t = predict(tweedie, X)
    @test length(preds_t) == 4

    tweedie_id = fit(TweedieRegression(power=0.0, link=:identity), X, y)
    preds_t_id = predict(tweedie_id, X)
    @test length(preds_t_id) == 4

    # Weighted fits
    poisson_w = fit(PoissonRegression(), X, y; weights=weights)
    @test length(predict(poisson_w, X)) == 4

    # Check errors
    @test_throws Tilia.InvalidHyperparameterError PoissonRegression(lambda=-1.0)
    @test_throws Tilia.InvalidHyperparameterError PoissonRegression(link=:unknown)
    @test_throws Tilia.InvalidHyperparameterError TweedieRegression(power=0.5)

    # Target constraints: Gamma requires positive values
    @test_throws Tilia.UnsupportedDataError fit(GammaRegression(), X, [2.0, 0.0, 3.0, 7.0])
    # Poisson requires nonnegative values
    @test_throws Tilia.UnsupportedDataError fit(PoissonRegression(), X, [2.0, -1.0, 3.0, 7.0])
end
