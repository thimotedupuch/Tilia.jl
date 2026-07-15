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

@testset "Classification curves and calibration" begin
    targets = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    roc = roc_curve(targets, scores)
    @test roc.thresholds == [Inf, 0.8, 0.4, 0.35, 0.1]
    @test roc.false_positive_rate == [0.0, 0.0, 0.5, 0.5, 1.0]
    @test roc.true_positive_rate == [0.0, 0.5, 0.5, 1.0, 1.0]
    @test area_under_curve(roc) == 0.75

    precision_recall = precision_recall_curve(targets, scores)
    @test precision_recall.thresholds == roc.thresholds
    @test precision_recall.recall == [0.0, 0.5, 0.5, 1.0, 1.0]
    @test precision_recall.precision ≈ [1.0, 1.0, 0.5, 2 / 3, 0.5]
    @test 0 <= area_under_curve(precision_recall) <= 1

    tied = roc_curve([0, 1, 0, 1], [0.7, 0.7, 0.2, 0.2])
    @test tied.thresholds == [Inf, 0.7, 0.2]
    @test tied.false_positive_rate == [0.0, 0.5, 1.0]
    @test tied.true_positive_rate == [0.0, 0.5, 1.0]

    weighted = roc_curve(targets, scores; weights=[1.0, 2.0, 3.0, 4.0])
    @test weighted.true_positive_rate[2] == 4 / 7
    @test weighted.false_positive_rate[3] == 2 / 3

    calibration = calibration_curve(targets, scores; n_bins=2)
    @test calibration.mean_predicted_probability ≈ [mean(scores[1:3]), 0.8]
    @test calibration.fraction_positive ≈ [1 / 3, 1.0]
    @test calibration.counts == [3.0, 1.0]
    quantile_calibration = calibration_curve(targets, scores; n_bins=2,
                                             strategy=:quantile)
    @test sum(quantile_calibration.counts) == length(targets)

    roc_curve(targets, scores)
    @test @allocated(roc_curve(targets, scores)) <= 20_000
    @test_throws ArgumentError roc_curve(fill(1, 4), scores)
    @test_throws ArgumentError calibration_curve(targets, [0.1, 1.2, 0.3, 0.4])
    @test_throws ArgumentError calibration_curve(targets, scores; strategy=:bad)
    @test_throws ArgumentError area_under_curve([1.0, 0.0], [0.0, 1.0])
end
