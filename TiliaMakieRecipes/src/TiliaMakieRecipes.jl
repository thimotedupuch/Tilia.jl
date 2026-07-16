module TiliaMakieRecipes

using Tilia
using Makie
using LinearAlgebra
import Makie: dendrogram

export TiliaPlot, tiliaplot, tiliaplot!
export ProjectionPlot, projectionplot, projectionplot!
export ScreePlot, screeplot, screeplot!
export Biplot, biplot, biplot!
export LoadingsPlot, loadingsplot, loadingsplot!
export ComponentPlot, componentplot, componentplot!
export ReconstructionPlot, reconstructionplot, reconstructionplot!
export clusterplot, dendrogram, decisionboundaryplot, treeplot
export learningcurveplot, validationcurveplot, residualplot
export predictedactualplot, residualdistributionplot, coefficientplot
export regularizationpathplot, mixturedensityplot, neighborhoodplot
export anomalyscoreplot, tuningheatmap, tuningparallelplot, modelcomparisonplot
export projectionplot3d, clusterplot3d, pointcloudplot
export neighborhoodplot3d, regressionsurfaceplot, tuninglandscapeplot
export partialdependenceplot, probabilitysimplexplot

Makie.@recipe TiliaPlot (result,) begin
    color = :steelblue
    linewidth = 2.5
    marker = :circle
    markersize = 10
    colormap = :Blues
    referencecolor = :gray55
    referencelinestyle = :dash
    show_reference = true
    show_values = true
    errorcolor = :steelblue
end

for Result in (Tilia.ConfusionMatrix, Tilia.ROCResult, Tilia.PrecisionRecallResult,
               Tilia.CalibrationResult, Tilia.PermutationImportanceResult,
               Tilia.CrossValidationResult, Tilia.OptimizationTrace)
    @eval Makie.plottype(::$Result) = TiliaPlot
end

_indices(values) = collect(eachindex(values))
_labels(values) = string.(values)
_mean(values) = sum(values) / length(values)
function _standard_deviation(values)
    length(values) < 2 && return zero(eltype(values))
    center = _mean(values)
    return sqrt(sum(x -> abs2(x - center), values) / (length(values) - 1))
end

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, result::Tilia.ConfusionMatrix)
    ticks = (_indices(result.labels), _labels(result.labels))
    return (title = "Confusion matrix", xlabel = "Predicted class",
            ylabel = "True class", xticks = ticks, yticks = ticks,
            yreversed = true, aspect = Makie.DataAspect())
end

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::Tilia.ROCResult) =
    (title = "ROC curve", xlabel = "False-positive rate",
     ylabel = "True-positive rate", limits = (-0.025, 1.025, -0.025, 1.025),
     xticks = 0:0.25:1, yticks = 0:0.25:1,
     aspect = Makie.DataAspect())

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::Tilia.PrecisionRecallResult) =
    (title = "Precision–recall curve", xlabel = "Recall", ylabel = "Precision",
     limits = (-0.025, 1.025, -0.025, 1.025),
     xticks = 0:0.25:1, yticks = 0:0.25:1, aspect = Makie.DataAspect())

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::Tilia.CalibrationResult) =
    (title = "Calibration curve", xlabel = "Mean predicted probability",
     ylabel = "Fraction positive", limits = (-0.025, 1.025, -0.025, 1.025),
     xticks = 0:0.25:1, yticks = 0:0.25:1,
     aspect = Makie.DataAspect())

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis},
                                         result::Tilia.PermutationImportanceResult)
    ticks = (_indices(result.feature_names), _labels(result.feature_names))
    return (title = "Permutation importance", xlabel = "Feature",
            ylabel = "Mean score decrease", xticks = ticks,
            xticklabelrotation = π / 4)
end

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis},
                                         result::Tilia.CrossValidationResult)
    return (title = "Cross-validation scores", xlabel = "Fold", ylabel = "Score",
            xticks = _indices(result.scores))
end

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis},
                                         result::Tilia.OptimizationTrace)
    status = result.converged ? "Converged" : "Did not converge"
    return (title = "Optimization trace", subtitle = status,
            xlabel = "Iteration", ylabel = "Objective")
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.ConfusionMatrix}})
    result = plot.result[]
    positions = _indices(result.labels)
    Makie.heatmap!(plot, positions, positions, result.matrix; colormap = plot.colormap)
    if plot.show_values[]
        points = [Makie.Point2f(j, i) for i in axes(result.matrix, 1)
                  for j in axes(result.matrix, 2)]
        values = [string(result.matrix[i, j]) for i in axes(result.matrix, 1)
                  for j in axes(result.matrix, 2)]
        Makie.text!(plot, points; text = values, align = (:center, :center),
                    color = :black)
    end
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.ROCResult}})
    result = plot.result[]
    if plot.show_reference[]
        Makie.ablines!(plot, 0, 1; color = plot.referencecolor,
                       linestyle = plot.referencelinestyle)
    end
    Makie.lines!(plot, result.false_positive_rate, result.true_positive_rate;
                 color = plot.color, linewidth = plot.linewidth, label = "ROC")
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.PrecisionRecallResult}})
    result = plot.result[]
    Makie.lines!(plot, result.recall, result.precision; color = plot.color,
                 linewidth = plot.linewidth, label = "Precision–recall")
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.CalibrationResult}})
    result = plot.result[]
    if plot.show_reference[]
        Makie.ablines!(plot, 0, 1; color = plot.referencecolor,
                       linestyle = plot.referencelinestyle, label = "Perfect calibration")
    end
    max_count = isempty(result.counts) ? 0 : maximum(result.counts)
    sizes = max_count == 0 ? plot.markersize[] :
            plot.markersize[] .* (0.5 .+ 0.5 .* sqrt.(result.counts ./ max_count))
    Makie.scatterlines!(plot, result.mean_predicted_probability, result.fraction_positive;
                        color = plot.color, linewidth = plot.linewidth,
                        marker = plot.marker, markersize = sizes, label = "Observed")
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.PermutationImportanceResult}})
    result = plot.result[]
    positions = _indices(result.mean_importance)
    Makie.barplot!(plot, positions, result.mean_importance; color = plot.color,
                   label = "Mean importance")
    Makie.errorbars!(plot, positions, result.mean_importance, result.standard_deviation;
                     color = plot.errorcolor, whiskerwidth = 8)
    if plot.show_reference[]
        Makie.hlines!(plot, [0]; color = plot.referencecolor,
                      linestyle = plot.referencelinestyle)
    end
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.CrossValidationResult}})
    result = plot.result[]
    positions = _indices(result.scores)
    Makie.scatterlines!(plot, positions, result.scores; color = plot.color,
                        linewidth = plot.linewidth, marker = plot.marker,
                        markersize = plot.markersize, label = "Fold score")
    if plot.show_reference[] && !isempty(result.scores)
        center = _mean(result.scores)
        spread = _standard_deviation(result.scores)
        Makie.band!(plot, positions, fill(center - spread, length(positions)),
                    fill(center + spread, length(positions)); color = (:gray, 0.15))
        Makie.hlines!(plot, [center]; color = plot.referencecolor,
                      linestyle = plot.referencelinestyle, label = "Mean")
    end
    return plot
end

function Makie.plot!(plot::TiliaPlot{<:Tuple{<:Tilia.OptimizationTrace}})
    result = plot.result[]
    positions = _indices(result.objective)
    Makie.scatterlines!(plot, positions, result.objective; color = plot.color,
                        linewidth = plot.linewidth, marker = plot.marker,
                        markersize = plot.markersize, label = "Objective")
    return plot
end

include("dimensionality_reduction.jl")
include("model_diagnostics.jl")
include("three_dimensional.jl")
include("explanatory_plots.jl")

end
