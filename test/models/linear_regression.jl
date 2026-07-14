@testset "Linear and ridge regression" begin
    X = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0]
    y = [2.0, 1.0, 3.0, 7.0]
    weights = [1.0, 2.0, 1.0, 3.0]

    linear = fit(LinearRegression(), X, y)
    @test linear.coefficients ≈ [1.2142857142857142, 1.2142857142857142]
    @test linear.intercept ≈ -1.6071428571428568
    @test predict(linear, X) ≈ [2.035714285714286, 0.8214285714285716,
                                3.25, 6.892857142857142]
    @test X' * (predict(linear, X) - y) ≈ zeros(2) atol=1e-12
    @test sum(predict(linear, X) - y) ≈ 0 atol=1e-12

    weighted = fit(LinearRegression(solver=:svd), X, y; weights=weights)
    @test weighted.coefficients ≈ [1.2366412213740472, 1.1984732824427473]
    @test weighted.intercept ≈ -1.5877862595419874

    ridge = fit(RidgeRegression(lambda=2.0), X, y)
    @test ridge.coefficients ≈ [0.9444444444444444, 0.9444444444444444]
    @test ridge.intercept ≈ -0.5277777777777772
    centered_X = X .- mean(X; dims=1)
    centered_y = y .- mean(y)
    @test centered_X' * (centered_X * ridge.coefficients - centered_y) +
          2ridge.coefficients ≈ zeros(2) atol=1e-12

    weighted_ridge = fit(RidgeRegression(lambda=2.0, solver=:svd), X, y; weights=weights)
    @test weighted_ridge.coefficients ≈ [1.0666666666666662, 1.1049645390070926]
    @test weighted_ridge.intercept ≈ -0.9418439716312053
    @test report(weighted_ridge).details.regularization == 2.0
    @test report(weighted_ridge).details.weighted

    no_intercept = fit(LinearRegression(fit_intercept=false), X, y)
    @test no_intercept.intercept == 0.0
    @test predict(no_intercept, X) ≈ X * no_intercept.coefficients

    for T in (Float32, Float64)
        fitted = fit(RidgeRegression(lambda=T(1)), T.(X), T.(y))
        @test eltype(fitted.coefficients) == T
        @test eltype(predict(fitted, T.(X))) == T
    end

    @test_throws Tilia.InvalidHyperparameterError RidgeRegression(lambda=-1)
    @test_throws Tilia.InvalidHyperparameterError LinearRegression(solver=:normal_equations)
    @test_throws Tilia.UnsupportedDataError fit(LinearRegression(), sparse(X), y)
    @test_throws Tilia.SchemaMismatchError predict(linear, ones(2, 3))
end
