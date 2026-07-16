ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1

using CairoMakie
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIRECTORY)

function save_diagnostic(filename, result; resolution = (760, 520), kwargs...)
    figure_axis_plot = plot(result; figure = (size = resolution,), kwargs...)
    save(joinpath(OUTPUT_DIRECTORY, filename), figure_axis_plot.figure; px_per_unit = 2)
    return figure_axis_plot
end

confusion = ConfusionMatrix(
    [86 5 1; 7 71 6; 0 8 76],
    ["setosa", "versicolor", "virginica"],
)

roc = ROCResult(
    [0.0, 0.01, 0.03, 0.08, 0.16, 0.31, 0.52, 0.74, 1.0],
    [0.0, 0.34, 0.58, 0.76, 0.87, 0.94, 0.98, 1.0, 1.0],
    [Inf, 0.93, 0.84, 0.72, 0.61, 0.49, 0.37, 0.20, -Inf],
)

precision_recall = PrecisionRecallResult(
    [1.0, 0.98, 0.95, 0.91, 0.87, 0.80, 0.71, 0.60, 0.48],
    [0.0, 0.18, 0.36, 0.55, 0.69, 0.81, 0.90, 0.96, 1.0],
    [Inf, 0.91, 0.82, 0.73, 0.64, 0.55, 0.43, 0.29, 0.10],
)

calibration = CalibrationResult(
    [0.05, 0.15, 0.26, 0.36, 0.47, 0.58, 0.68, 0.79, 0.89, 0.96],
    [0.03, 0.11, 0.22, 0.34, 0.45, 0.61, 0.72, 0.83, 0.91, 0.98],
    [120.0, 105.0, 94.0, 83.0, 70.0, 61.0, 49.0, 37.0, 25.0, 12.0],
    collect(0.0:0.1:1.0),
)

importance = PermutationImportanceResult(
    0.91,
    [0.21 0.24 0.22 0.25 0.23;
     0.14 0.12 0.16 0.15 0.13;
     0.08 0.07 0.09 0.06 0.08;
     0.03 0.04 0.02 0.03 0.04;
     -0.01 0.00 0.01 -0.01 0.00],
    [0.23, 0.14, 0.076, 0.032, -0.002],
    [0.016, 0.016, 0.011, 0.008, 0.008],
    ["petal length", "petal width", "sepal length", "sepal width", "noise"],
)

cross_validation = CrossValidationResult(
    [0.86, 0.91, 0.88, 0.93, 0.89, 0.92, 0.87, 0.90],
    Any[], Any[], [Int[] for _ in 1:8], [Int[] for _ in 1:8],
)

optimization = OptimizationTrace(
    [2.80, 1.92, 1.41, 1.08, 0.86, 0.70, 0.59, 0.51, 0.46, 0.43, 0.41],
    true,
)

save_diagnostic("confusion_matrix.png", confusion; colormap = :Blues)
save_diagnostic("roc_curve.png", roc; color = :darkorange)
save_diagnostic("precision_recall_curve.png", precision_recall; color = :seagreen)
save_diagnostic("calibration_curve.png", calibration; color = :mediumpurple)
save_diagnostic("permutation_importance.png", importance; color = :cornflowerblue)
save_diagnostic("cross_validation.png", cross_validation; color = :teal)
save_diagnostic("optimization_trace.png", optimization; color = :crimson)

dashboard = Figure(size = (1500, 1350), fontsize = 17)

axis_confusion, _ = plot(dashboard[1, 1], confusion; colormap = :Blues)
axis_roc, _ = plot(dashboard[1, 2], roc; color = :darkorange)
axis_pr, _ = plot(dashboard[2, 1], precision_recall; color = :seagreen)
axis_calibration, _ = plot(dashboard[2, 2], calibration; color = :mediumpurple)
axis_importance, _ = plot(dashboard[3, 1], importance; color = :cornflowerblue)
axis_cv, _ = plot(dashboard[3, 2], cross_validation; color = :teal)
axis_optimization, _ = plot(dashboard[4, 1:2], optimization; color = :crimson)

Label(dashboard[0, 1:2], "Tilia diagnostic plots", fontsize = 30, font = :bold)
save(joinpath(OUTPUT_DIRECTORY, "diagnostic_dashboard.png"), dashboard; px_per_unit = 1.5)

println("Generated figures in: ", OUTPUT_DIRECTORY)
foreach(name -> println("  ", name), sort(readdir(OUTPUT_DIRECTORY)))
