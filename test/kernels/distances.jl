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
