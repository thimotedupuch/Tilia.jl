@testset "Partial least-squares closed-form reference" begin
    X = Float64[1 1 2; 2 2 4; 3 3 6; 4 4 8; 5 5 10; 6 6 12]
    y = 3 .* X[:, 1] .- 2
    fitted = fit(PartialLeastSquaresRegression(n_components=1), X, y)
    @test predict(fitted, X) ≈ y atol=1e-10
    @test fitted.intercept + sum(fitted.coefficients .* X[1, :]) ≈ y[1] atol=1e-10
end
