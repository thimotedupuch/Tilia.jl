using Tilia.Kernels

@testset "Loss and sparse kernels" begin
    targets = [1.0, 2.0, 3.0]
    predictions = [1.0, 4.0, 2.0]
    @test mean_squared_error(targets, predictions) == 5 / 3
    @test root_mean_squared_error(targets, predictions) ≈ sqrt(5 / 3)
    @test mean_squared_error(targets, predictions; weights=[1.0, 0.0, 1.0]) == 0.5
    @test mean_absolute_error(targets, predictions) ≈ 1.0
    @test mean_absolute_error(targets, predictions; weights=[1.0, 0.0, 1.0]) ≈ 0.5
    @test huber_loss(targets, predictions; delta=1.5) ≈ (0.0 + 1.875 + 0.5) / 3
    @test quantile_loss(targets, predictions; quantile=0.75) ≈ (0.0 + 0.5 + 0.75) / 3
    @test_throws ArgumentError huber_loss(targets, predictions; delta=-1.0)
    @test_throws ArgumentError quantile_loss(targets, predictions; quantile=1.5)

    probabilities = [0.8 0.2; 0.25 0.75]
    @test log_loss([1, 2], probabilities) ≈ -mean(log.([0.8, 0.75]))
    @test_throws ArgumentError log_loss([1, 2], [0.8 0.3; 0.2 0.8])

    sparse_matrix = sparse([1.0 0.0; 2.0 3.0])
    original_pattern = copy(sparse_matrix.colptr), copy(sparse_matrix.rowval)
    scaled = scale_columns(sparse_matrix, [2.0, 4.0])
    @test Matrix(scaled) == [2.0 0.0; 4.0 12.0]
    @test Matrix(sparse_matrix) == [1.0 0.0; 2.0 3.0]
    @test scaled.colptr == original_pattern[1]
    @test scaled.rowval == original_pattern[2]
    @test scale_columns!(sparse_matrix, [2.0, 4.0]) === sparse_matrix
end
