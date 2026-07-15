using Tilia.Kernels

@testset "Kernel types and allocations" begin
    values = Float64[1, 2, 3, 4]
    weights = Float64[1, 1, 2, 2]
    @test @inferred(weighted_mean(values, weights)) isa Float64
    @test @inferred(reduction_mean(values)) isa Float64
    @test @inferred(extrema_values(values)) == (1.0, 4.0)
    @test @inferred(stable_norm(values)) isa Float64
    @test @inferred(sigmoid(1.0)) isa Float64
    @test @inferred(euclidean(values, weights)) isa Float64

    destination = sparse([1.0 0.0; 0.0 2.0])
    scales = [2.0, 3.0]
    scale_columns!(destination, scales) # compile before measuring steady state
    allocations = @allocated scale_columns!(destination, scales)
    @test allocations <= 128

    reduction_sum(values)
    @test @allocated(reduction_sum(values)) <= 64

    logits = randn(32, 4)
    softmax(logits; dims=2)
    allocations = @allocated softmax(logits; dims=2)
    @test allocations <= 20_000
end
