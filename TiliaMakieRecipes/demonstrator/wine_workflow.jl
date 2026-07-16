ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1

using CairoMakie
using DataFrames
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output", "wine_workflow")
mkpath(OUTPUT_DIRECTORY)

# MLDatasets stores tabular observations in columns when `as_df=false`; Tilia
# uses rows as observations, so transpose both arrays into that convention.
wine = Wine(as_df=false)
X = Matrix(transpose(wine.features))
y = vec(wine.targets)
feature_names = string.(wine.metadata["feature_names"])

X_train, X_test, y_train, y_test, _, _ = train_test_split(
    X, y; test_size=0.30, seed=42, stratify=y,
)

model = Chain(Standardize(), LogisticRegression(
    lambda=0.1, max_iterations=100, tolerance=1.0e-8,
))
fitted = fit(model, X_train, y_train)
predictions = predict(fitted, X_test)
probabilities = predict_proba(fitted, X_test)

classes = sort(unique(y_train))
class_rocs = [roc_curve(y_test .== class, probabilities[:, column]; positive_label=true)
              for (column, class) in enumerate(classes)]
positive_column = argmin(area_under_curve.(class_rocs))
positive_class = classes[positive_column]
positive_scores = probabilities[:, positive_column]
binary_targets = y_test .== positive_class

confusion = confusion_matrix(y_test, predictions; labels=classes)
roc = class_rocs[positive_column]
precision_recall = precision_recall_curve(
    binary_targets, positive_scores; positive_label=true,
)
calibration = calibration_curve(
    binary_targets, positive_scores; positive_label=true,
    n_bins=6, strategy=:quantile,
)

raw_importance = permutation_importance(
    fitted, X_test, y_test; n_repeats=20,
    context=FitContext(seed=42),
)
importance = PermutationImportanceResult(
    raw_importance.baseline_score,
    raw_importance.importances,
    raw_importance.mean_importance,
    raw_importance.standard_deviation,
    feature_names,
)

cross_validation = cross_validate(
    model, X, y; cv=KFold(8; shuffle=true, seed=42),
)

# LogisticRegression records one Newton objective trace per one-vs-rest class.
classifier_report = report(last(fitted.fitted_nodes))
class_trace = classifier_report.details.objective_history[positive_column]
class_converged = classifier_report.details.convergence[positive_column]
optimization = OptimizationTrace(class_trace, class_converged)

diagnostics = (
    ("wine_confusion_matrix.png", confusion, :Blues),
    ("wine_roc_curve.png", roc, :darkorange),
    ("wine_precision_recall_curve.png", precision_recall, :seagreen),
    ("wine_calibration_curve.png", calibration, :mediumpurple),
    ("wine_permutation_importance.png", importance, :cornflowerblue),
    ("wine_cross_validation.png", cross_validation, :teal),
    ("wine_optimization_trace.png", optimization, :crimson),
)

for (filename, diagnostic, color) in diagnostics
    figure_axis_plot = diagnostic isa ConfusionMatrix ?
        plot(diagnostic; figure=(size=(850, 600),), colormap=color) :
        plot(diagnostic; figure=(size=(850, 600),), color=color)
    save(joinpath(OUTPUT_DIRECTORY, filename), figure_axis_plot.figure; px_per_unit=2)
end

dashboard = Figure(size=(1600, 1450), fontsize=17)
plot(dashboard[1, 1], confusion; colormap=:Blues)
plot(dashboard[1, 2], roc; color=:darkorange,
     axis=(subtitle="Class $(positive_class) vs. rest",))
plot(dashboard[2, 1], precision_recall; color=:seagreen,
     axis=(subtitle="Class $(positive_class) vs. rest",))
plot(dashboard[2, 2], calibration; color=:mediumpurple,
     axis=(subtitle="Class $(positive_class) vs. rest",))
plot(dashboard[3, 1], importance; color=:cornflowerblue,
     axis=(xticklabelsize=11,))
plot(dashboard[3, 2], cross_validation; color=:teal)
plot(dashboard[4, 1:2], optimization; color=:crimson,
     axis=(subtitle="Newton solver · class $(positive_class) vs. rest · " *
                    (class_converged ? "converged" : "not converged"),))

accuracy = accuracy_score(y_test, predictions)
auc = area_under_curve(roc)
Label(dashboard[0, 1:2], "Tilia workflow on MLDatasets.Wine", fontsize=30, font=:bold)
Label(dashboard[5, 1:2],
      "$(length(y_train)) train / $(length(y_test)) test observations · " *
      "test accuracy=$(round(accuracy; digits=3)) · class $(positive_class) AUC=$(round(auc; digits=3))",
      fontsize=18)
save(joinpath(OUTPUT_DIRECTORY, "wine_workflow_dashboard.png"), dashboard; px_per_unit=1.5)

println("Dataset: MLDatasets.Wine")
println("Training observations: ", length(y_train))
println("Test observations: ", length(y_test))
println("Test accuracy: ", round(accuracy; digits=4))
println("Class ", positive_class, " one-vs-rest AUC: ", round(auc; digits=4))
println("Mean cross-validation accuracy: ",
        round(sum(cross_validation.scores) / length(cross_validation.scores); digits=4))
println("Generated figures in: ", OUTPUT_DIRECTORY)
