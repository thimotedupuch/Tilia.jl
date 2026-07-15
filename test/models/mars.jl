@testset "Adaptive regression splines" begin
    x = collect(range(-2.0, 2.0; length=80))
    X = reshape(x, :, 1)
    y = 1 .+ 2 .* max.(x .- 0.3, 0) .- 1.5 .* max.(-0.5 .- x, 0)
    fitted = fit(MARSRegressor(max_terms=9, max_knots=20), X, y)
    @test root_mean_squared_error(y, predict(fitted, X)) < 0.02
    @test first(fitted.terms).factors == Tilia.HingeFactor{Float64}[]
    @test report(fitted).details.solver == :qr
    @test report(fitted).details.selected_terms <= 9

    grid = collect(range(-1.0, 1.0; length=10))
    X2 = hcat(repeat(grid; inner=length(grid)),
              repeat(grid; outer=length(grid)))
    y2 = max.(X2[:, 1], 0) .* max.(X2[:, 2], 0) .+ 0.2 .* X2[:, 1]
    interaction = fit(MARSRegressor(max_terms=11, max_degree=2, max_knots=10), X2, y2)
    @test root_mean_squared_error(y2, predict(interaction, X2)) < 0.08
    @test maximum(length(term.factors) for term in interaction.terms) <= 2
    @test_throws Tilia.InvalidHyperparameterError MARSRegressor(max_terms=2)
    @test_throws Tilia.InvalidHyperparameterError MARSRegressor(max_degree=0)
    @test_throws Tilia.SchemaMismatchError predict(fitted, ones(2, 2))
end
