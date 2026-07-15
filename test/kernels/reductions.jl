using Tilia.Kernels
using SparseArrays

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
    @test reduction_sum(Float32[1, 2, 3]) === 6.0f0
    @test reduction_mean(Float32[1, 2, 3]) === 2.0f0
    @test extrema_values([3, 1, 4]) == (1, 4)
    @test argmin_index([2, 1, 1]) == 2
    @test argmax_index([3, 3, 1]) == 1
    @test stable_sum([1e16, 1.0, -1e16]) == 1.0
    @test weighted_sum([1e16, 1.0, -1e16], ones(3); stable=true) == 1.0
    @test weighted_mean(Float32[1, 2], Float32[1, 1];
        accumulation_type=Float64) === 1.5
    @test_throws DimensionMismatch weighted_mean(values, weights[1:2])
    @test_throws ArgumentError weighted_mean(values, zeros(3))
    @test_throws ArgumentError weighted_variance([1.0], [1.0]; corrected=true)
    @test_throws ArgumentError reduction_mean(Float64[])
    @test_throws ArgumentError extrema_values(Int[])
end

@testset "Numerics policy" begin
    policy = NumericsPolicy(Float32; accumulation_type=Float64,
        tolerance_scale=2, max_iterations=25, stable_summation=false,
        missing_policy=:allow, finite_policy=:allow,
        overflow_policy=:saturate, underflow_policy=:flush_zero,
        deterministic_reductions=false, sparse_centering=:densify)
    context = FitContext(numerics=policy)
    fitted = fit(MeanRegressor(), Float32[1; 2;;], Float32[1, 2]; context)
    summary = report(fitted).details.numerical_policy
    @test summary.float_type == "Float32"
    @test summary.accumulation_type == "Float64"
    @test summary.max_iterations == 25
    @test summary.missing_policy == :allow
    @test summary.underflow_policy == :flush_zero
    @test summary.sparse_centering == :densify
    @test Tilia.effective_tolerance(context, 0.25f0) == 0.5f0
    @test Tilia.effective_max_iterations(context, 100) == 25
    sparse = SparseArrays.sparse(Float32[1 0; 3 2])
    centered = fit(Standardize(), sparse; context)
    @test transform(centered, sparse) ≈ [-1 -1; 1 1]
    @test_throws Tilia.InvalidHyperparameterError NumericsPolicy(; sparse_centering=:invalid)
    @test_throws Tilia.InvalidHyperparameterError NumericsPolicy(; missing_policy=:invalid)
    @test_throws Tilia.InvalidHyperparameterError NumericsPolicy(; underflow_policy=:invalid)
end
