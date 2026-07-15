@testset "Random projection moment reference" begin
    X = Matrix{Float64}(I, 80, 80)
    fitted = fit(RandomProjection(n_components=40), X;
                 context=FitContext(seed=2026))
    squared_column_norms = vec(sum(abs2, fitted.projection; dims=1))
    @test abs(mean(squared_column_norms) - 1) < 0.08
    @test abs(mean(fitted.projection)) < 0.04
end
