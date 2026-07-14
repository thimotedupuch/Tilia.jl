using Test
using Tilia
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

    cross_validation = CrossValidationResult(
        [0.8, 0.9, 0.85], Any[], Any[], [Int[] for _ in 1:3], [Int[] for _ in 1:3])
    @test Makie.plottype(cross_validation) == Makie.Scatter
    @test Makie.convert_arguments(Makie.Scatter, cross_validation) ==
          ([1, 2, 3], cross_validation.scores)

    optimization = OptimizationTrace([3.0, 2.0, 1.5], true)
    @test Makie.plottype(optimization) == Makie.Lines
    @test Makie.convert_arguments(Makie.Lines, optimization) ==
          ([1, 2, 3], optimization.objective)

    # Exercise Makie's complete generic plotting dispatch without a display backend.
    plot = Makie.plot(confusion)
    @test plot isa Makie.FigureAxisPlot
end
