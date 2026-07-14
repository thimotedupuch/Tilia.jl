@testset "Regression and classification metrics" begin
    targets = [:cat, :cat, :dog, :dog, :bird, :bird]
    predictions = [:cat, :dog, :dog, :dog, :bird, :cat]
    @test accuracy_score(targets, predictions) == 4 / 6
    matrix = confusion_matrix(targets, predictions)
    @test matrix.labels == [:bird, :cat, :dog]
    @test matrix.matrix == [1 1 0; 0 1 1; 0 0 2]
    @test sum(matrix.matrix) == length(targets)
    @test precision_score(targets, predictions; average=:none) ≈ [1.0, 0.5, 2 / 3]
    @test recall_score(targets, predictions; average=:none) ≈ [0.5, 0.5, 1.0]
    @test f1_score(targets, predictions; average=:none) ≈ [2 / 3, 0.5, 0.8]
    @test precision_score(targets, predictions; average=:micro) == accuracy_score(targets, predictions)
    @test recall_score(targets, predictions; average=:micro) == accuracy_score(targets, predictions)
    @test f1_score(targets, predictions; average=:micro) ≈ accuracy_score(targets, predictions)
    @test accuracy_score(targets, predictions; weights=[1, 1, 1, 1, 2, 2]) == 5 / 8

    probabilities = [0.8 0.2; 0.4 0.6; 0.1 0.9]
    @test log_loss([:negative, :positive, :positive], probabilities;
                   labels=[:negative, :positive]) ≈ -mean(log.([0.8, 0.6, 0.9]))
    @test mean_squared_error([1.0, 2.0], [2.0, 4.0]) == 2.5
    @test root_mean_squared_error([1.0, 2.0], [2.0, 4.0]) == sqrt(2.5)
    @test_throws DimensionMismatch accuracy_score([1], [1, 2])
    @test_throws ArgumentError precision_score(targets, predictions; average=:binary)
end
