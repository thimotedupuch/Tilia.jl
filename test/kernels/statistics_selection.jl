@testset "Statistical and selection kernels" begin
    X = [1.0 2; 2 4; 4 8]
    @test Tilia.Kernels.covariance_matrix(X) ≈ cov(X; dims=1)
    weights = [1.0, 2, 1]
    weighted = Tilia.Kernels.weighted_covariance(X, weights)
    average = vec(sum(X .* weights; dims=1)) / sum(weights)
    expected = transpose(X .- transpose(average)) *
               ((X .- transpose(average)) .* weights) / sum(weights)
    @test weighted ≈ expected

    contingency, rows, columns = Tilia.Kernels.contingency_matrix(
        [:a, :a, :b, :b, :b], [1, 2, 1, 1, 2])
    @test rows == [:a, :b]
    @test columns == [1, 2]
    @test contingency == [1 1; 2 1]
    @test Tilia.Kernels.class_counts([:b, :a, :b]) == ([:a, :b], [1, 2])
    @test Tilia.Kernels.histogram_counts([0.0, 0.5, 1.0, 2.0], [0.0, 1.0, 2.0]) == [2, 2]

    values = [3.0, 1.0, 3.0, 2.0]
    @test Tilia.Kernels.topk_indices(values, 2) == [1, 3]
    @test Tilia.Kernels.topk_indices(values, 2; largest=false) == [2, 4]
    @test Tilia.Kernels.quantile_value([1.0, 2, 3, 4], 0.5) == 2.5
    @test Tilia.Kernels.rank_values(values) == [3.5, 1.0, 3.5, 2.0]
    @test Tilia.Kernels.rank_values(values; ties=:dense) == [3.0, 1.0, 3.0, 2.0]

    precision = [2.0 0; 0 0.5]
    @test Tilia.Kernels.mahalanobis_distance([1.0, 2], [0.0, 0], precision) == 2.0

    sparse_X = sparse([1.0 0; 2 3])
    @test Tilia.Kernels.sparse_column_sums(sparse_X) == [3.0, 3.0]
    @test Tilia.Kernels.sparse_matvec(sparse_X, [2.0, 1]) == [2.0, 7.0]
    @test Tilia.Kernels.sparse_dot(sparsevec([1, 3], [2.0, 4], 3), [1.0, 2, 3]) == 14
    @test Tilia.Kernels.center_sparse(sparse_X, zeros(2)) == sparse_X
    @test_throws ArgumentError Tilia.Kernels.center_sparse(sparse_X, [1.0, 0])
    @test Tilia.Kernels.center_sparse(sparse_X, [1.0, 0]; allow_densify=true) ==
          [-0.0 0.0; 1.0 3.0]
end
