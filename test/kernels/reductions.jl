using Tilia.Kernels

@testset "Reduction kernels" begin
    values = Float64[1, 2, 5]
    weights = Float64[1, 2, 1]
    @test weighted_sum(values, weights) == 10
    @test weighted_mean(values, weights) == 2.5
    @test weighted_variance(values, weights) == 2.25
    @test weighted_variance(values, weights; corrected=true) ≈ 3.6
    @test stable_norm([3.0, 4.0]) == 5.0
    @test isfinite(stable_norm([1e300, 1e300]))
    @test stable_norm(Float32[]) === 0.0f0
    @test_throws DimensionMismatch weighted_mean(values, weights[1:2])
    @test_throws ArgumentError weighted_mean(values, zeros(3))
    @test_throws ArgumentError weighted_variance([1.0], [1.0]; corrected=true)
end
