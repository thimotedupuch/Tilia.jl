module TiliaMakieRecipes

using Tilia
using Makie

Makie.plottype(::Tilia.ConfusionMatrix) = Makie.Heatmap
Makie.convert_arguments(::Type{<:Makie.Heatmap}, result::Tilia.ConfusionMatrix) =
    (collect(axes(result.matrix, 1)), collect(axes(result.matrix, 2)), result.matrix)

Makie.plottype(::Tilia.ROCResult) = Makie.Lines
Makie.convert_arguments(::Type{<:Makie.Lines}, result::Tilia.ROCResult) =
    (result.false_positive_rate, result.true_positive_rate)

Makie.plottype(::Tilia.PrecisionRecallResult) = Makie.Lines
Makie.convert_arguments(::Type{<:Makie.Lines}, result::Tilia.PrecisionRecallResult) =
    (result.recall, result.precision)

Makie.plottype(::Tilia.CalibrationResult) = Makie.Scatter
Makie.convert_arguments(::Type{<:Makie.Scatter}, result::Tilia.CalibrationResult) =
    (result.mean_predicted_probability, result.fraction_positive)

Makie.plottype(::Tilia.PermutationImportanceResult) = Makie.BarPlot
Makie.convert_arguments(::Type{<:Makie.BarPlot}, result::Tilia.PermutationImportanceResult) =
    (collect(eachindex(result.mean_importance)), result.mean_importance)

Makie.plottype(::Tilia.CrossValidationResult) = Makie.Scatter
Makie.convert_arguments(::Type{<:Makie.Scatter}, result::Tilia.CrossValidationResult) =
    (collect(eachindex(result.scores)), result.scores)

Makie.plottype(::Tilia.OptimizationTrace) = Makie.Lines
Makie.convert_arguments(::Type{<:Makie.Lines}, result::Tilia.OptimizationTrace) =
    (collect(eachindex(result.objective)), result.objective)

end
