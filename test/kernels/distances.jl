using Tilia.Kernels

@testset "Distance kernels" begin
    left = [0.0, 0.0]
    right = [3.0, 4.0]
    @test squared_euclidean(left, right) == 25
    @test euclidean(left, right) == 5
    @test manhattan(left, right) == 7
    @test cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1
    X = [0.0 0.0; 3.0 4.0]
    @test pairwise_distances(X) ≈ [0.0 5.0; 5.0 0.0]
    @test pairwise_distances(X; metric=:squared_euclidean) ≈ [0.0 25.0; 25.0 0.0]
    @test pairwise_distances(Float32.(X)) isa Matrix{Float32}
    @test_throws DimensionMismatch euclidean([1.0], [1.0, 2.0])
    @test_throws ArgumentError cosine_distance([0.0], [1.0])
end

@testset "Blocked and threaded pairwise distances" begin
    X = Float32[sin(i * j) for i in 1:17, j in 1:5]
    Y = Float32[cos(i + j) for i in 1:9, j in 1:5]
    expected = Tilia.Kernels.pairwise_distances(X, Y)
    blocked = Tilia.Kernels.pairwise_distance_blocks(X, Y; block_size=4)
    threaded = Tilia.Kernels.pairwise_distance_blocks(
        X, Y; block_size=3, threaded=true)
    @test blocked ≈ expected
    @test threaded ≈ expected
    @test eltype(threaded) == Float32
    @test_throws ArgumentError Tilia.Kernels.pairwise_distance_blocks(X, Y; block_size=0)
end

@testset "BLAS-backed Euclidean distance matrix" begin
    left = Float32[1 2; 3 4; -1 0]
    right = Float32[0 1; 2 2]
    expected_squared = Float32[
        sum(abs2, left[1, :] .- right[1, :]) sum(abs2, left[1, :] .- right[2, :]);
        sum(abs2, left[2, :] .- right[1, :]) sum(abs2, left[2, :] .- right[2, :]);
        sum(abs2, left[3, :] .- right[1, :]) sum(abs2, left[3, :] .- right[2, :])
    ]
    squared = Tilia.Kernels.pairwise_distances(left, right; metric=:squared_euclidean)
    @test squared ≈ expected_squared
    @test Tilia.Kernels.pairwise_distances(left, right) ≈ sqrt.(expected_squared)
    @test eltype(squared) == Float32
    @test all(>=(0), Tilia.Kernels.pairwise_distances(left; metric=:squared_euclidean))

    integer_distances = Tilia.Kernels.pairwise_distances(Int[1 2; 3 4])
    @test eltype(integer_distances) <: AbstractFloat
    @test diag(integer_distances) == zeros(2)
end
