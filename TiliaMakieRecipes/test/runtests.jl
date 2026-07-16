using Test
using Tilia
using TiliaMakieRecipes
using Makie

@testset "Makie semantic result conversions" begin
    confusion = ConfusionMatrix([4 1; 2 5], [:negative, :positive])
    @test Makie.plottype(confusion) == Makie.Heatmap
    @test Makie.convert_arguments(Makie.Heatmap, confusion) ==
          ([1, 2], [1, 2], confusion.matrix)

    roc = ROCResult([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], [Inf, 0.5, -Inf])
    @test Makie.plottype(roc) == Makie.Lines
    @test Makie.convert_arguments(Makie.Lines, roc) ==
          (roc.false_positive_rate, roc.true_positive_rate)

    precision_recall = PrecisionRecallResult([1.0, 0.5], [0.0, 1.0], [Inf, 0.2])
    @test Makie.plottype(precision_recall) == Makie.Lines
    @test Makie.convert_arguments(Makie.Lines, precision_recall) ==
          (precision_recall.recall, precision_recall.precision)

    calibration = CalibrationResult([0.2, 0.8], [0.1, 0.9], [5.0, 5.0],
                                    [0.0, 0.5, 1.0])
    @test Makie.plottype(calibration) == Makie.Scatter
    @test Makie.convert_arguments(Makie.Scatter, calibration) ==
          (calibration.mean_predicted_probability, calibration.fraction_positive)

    importance = PermutationImportanceResult(
        0.9, [0.2 0.3; 0.0 0.1], [0.25, 0.05], [0.05, 0.05], [:x1, :x2])
    @test Makie.plottype(importance) == Makie.BarPlot
    @test Makie.convert_arguments(Makie.BarPlot, importance) ==
          ([1, 2], importance.mean_importance)

    cross_validation = CrossValidationResult(
        [0.8, 0.9, 0.85], Any[], Any[], [Int[] for _ in 1:3], [Int[] for _ in 1:3])
    @test Makie.plottype(cross_validation) == Makie.Scatter
    @test Makie.convert_arguments(Makie.Scatter, cross_validation) ==
          ([1, 2, 3], cross_validation.scores)

    optimization = OptimizationTrace([3.0, 2.0, 1.5], true)
    @test Makie.plottype(optimization) == Makie.Lines
    @test Makie.convert_arguments(Makie.Lines, optimization) ==
          ([1, 2, 3], optimization.objective)

    plot = Makie.plot(confusion)
    @test plot isa Makie.FigureAxisPlot
end
