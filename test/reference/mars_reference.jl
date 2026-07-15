@testset "MARS hinge-function numerical reference" begin
    x = collect(range(-3.0, 3.0; length=121))
    X = reshape(x, :, 1)
    exact = 0.7 .+ 1.2 .* max.(x .- 0.5, 0) .- 0.8 .* max.(-1.0 .- x, 0)
    fitted = fit(MARSRegressor(max_terms=13, max_knots=60,
                               pruning_penalty=2.0), X, exact)
    @test root_mean_squared_error(exact, predict(fitted, X)) < 1e-2
    @test report(fitted).details.residual_sum_squares < 2e-2
end
