ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CairoMakie
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output", "mnist_workflow")
mkpath(OUTPUT_DIRECTORY)

function balanced_indices(labels, observations_per_class)
    classes = sort(unique(labels))
    return reduce(vcat, findall(==(class), labels)[1:observations_per_class]
                  for class in classes)
end

function observation_matrix(images, indices)
    selected = images[:, :, indices]
    return permutedims(reshape(selected, :, length(indices)))
end

mnist_train = MNIST(Float32, :train)
mnist_test = MNIST(Float32, :test)

# Balanced subsets keep this demonstrator practical while retaining all ten
# classes and thousands of real observations.
train_indices = balanced_indices(mnist_train.targets, 500)
test_indices = balanced_indices(mnist_test.targets, 200)
cv_indices = balanced_indices(mnist_train.targets, 300)

X_train_pixels = observation_matrix(mnist_train.features, train_indices)
y_train = mnist_train.targets[train_indices]
X_test_pixels = observation_matrix(mnist_test.features, test_indices)
y_test = mnist_test.targets[test_indices]
X_cv_pixels = observation_matrix(mnist_train.features, cv_indices)
y_cv = mnist_train.targets[cv_indices]

# Fit dimensionality reduction only on training observations, then train the
# classifier on the resulting representation.
pca = fit(PCA(n_components=35), X_train_pixels)
X_train = transform(pca, X_train_pixels)
X_test = transform(pca, X_test_pixels)

classifier = Chain(Standardize(), LogisticRegression(
    lambda=0.2, max_iterations=100, tolerance=1.0e-7,
))
fitted = fit(classifier, X_train, y_train)
predictions = predict(fitted, X_test)
probabilities = predict_proba(fitted, X_test)

classes = sort(unique(y_train))
confusion = confusion_matrix(y_test, predictions; labels=classes)

# Use the genuinely hardest held-out one-vs-rest class for informative curves.
class_rocs = [roc_curve(y_test .== class, probabilities[:, column]; positive_label=true)
              for (column, class) in enumerate(classes)]
positive_column = argmin(area_under_curve.(class_rocs))
positive_class = classes[positive_column]
positive_scores = probabilities[:, positive_column]
binary_targets = y_test .== positive_class
roc = class_rocs[positive_column]
precision_recall = precision_recall_curve(
    binary_targets, positive_scores; positive_label=true,
)
calibration = calibration_curve(
    binary_targets, positive_scores; positive_label=true,
    n_bins=10, strategy=:quantile,
)

raw_importance = permutation_importance(
    fitted, X_test, y_test; n_repeats=10,
    context=FitContext(seed=42),
)
top_components = sortperm(abs.(raw_importance.mean_importance); rev=true)[1:15]
importance = PermutationImportanceResult(
    raw_importance.baseline_score,
    raw_importance.importances[top_components, :],
    raw_importance.mean_importance[top_components],
    raw_importance.standard_deviation[top_components],
    ["PC$(component)" for component in top_components],
)

# PCA remains inside each fold, so cross-validation never learns its projection
# from held-out fold observations.
cv_model = Chain(
    PCA(n_components=35), Standardize(),
    LogisticRegression(lambda=0.2, max_iterations=100, tolerance=1.0e-7),
)
cross_validation = cross_validate(
    cv_model, X_cv_pixels, y_cv; cv=KFold(5; shuffle=true, seed=42),
)

classifier_report = report(last(fitted.fitted_nodes))
class_trace = classifier_report.details.objective_history[positive_column]
class_converged = classifier_report.details.convergence[positive_column]
optimization = OptimizationTrace(class_trace, class_converged)

diagnostics = (
    ("mnist_confusion_matrix.png", confusion, :Blues),
    ("mnist_roc_curve.png", roc, :darkorange),
    ("mnist_precision_recall_curve.png", precision_recall, :seagreen),
    ("mnist_calibration_curve.png", calibration, :mediumpurple),
    ("mnist_permutation_importance.png", importance, :cornflowerblue),
    ("mnist_cross_validation.png", cross_validation, :teal),
    ("mnist_optimization_trace.png", optimization, :crimson),
)

for (filename, diagnostic, color) in diagnostics
    figure_axis_plot = diagnostic isa ConfusionMatrix ?
        plot(diagnostic; figure=(size=(1000, 800),), colormap=color,
             axis=(xticklabelsize=14, yticklabelsize=14)) :
        plot(diagnostic; figure=(size=(900, 650),), color=color)
    save(joinpath(OUTPUT_DIRECTORY, filename), figure_axis_plot.figure; px_per_unit=2)
end

dashboard = Figure(size=(1800, 1600), fontsize=17)
plot(dashboard[1, 1], confusion; colormap=:Blues, show_values=false)
plot(dashboard[1, 2], roc; color=:darkorange,
     axis=(subtitle="Digit $(positive_class) vs. rest",))
plot(dashboard[2, 1], precision_recall; color=:seagreen,
     axis=(subtitle="Digit $(positive_class) vs. rest",))
plot(dashboard[2, 2], calibration; color=:mediumpurple,
     axis=(subtitle="Digit $(positive_class) vs. rest",))
plot(dashboard[3, 1], importance; color=:cornflowerblue,
     axis=(xticklabelsize=11,))
plot(dashboard[3, 2], cross_validation; color=:teal)
plot(dashboard[4, 1:2], optimization; color=:crimson,
     axis=(subtitle="Newton solver · digit $(positive_class) vs. rest · " *
                    (class_converged ? "converged" : "not converged"),))

accuracy = accuracy_score(y_test, predictions)
auc = area_under_curve(roc)
mean_cv = sum(cross_validation.scores) / length(cross_validation.scores)
Label(dashboard[0, 1:2], "Tilia workflow on MLDatasets.MNIST", fontsize=30, font=:bold)
Label(dashboard[5, 1:2],
      "$(length(y_train)) train / $(length(y_test)) test observations · " *
      "35 PCA components · test accuracy=$(round(accuracy; digits=3)) · " *
      "digit $(positive_class) AUC=$(round(auc; digits=3))",
      fontsize=18)
save(joinpath(OUTPUT_DIRECTORY, "mnist_workflow_dashboard.png"), dashboard; px_per_unit=1.5)

println("Dataset: MLDatasets.MNIST")
println("Training observations: ", length(y_train))
println("Test observations: ", length(y_test))
println("Classes: ", length(classes))
println("Test accuracy: ", round(accuracy; digits=4))
println("Hardest digit: ", positive_class)
println("Hardest one-vs-rest AUC: ", round(auc; digits=4))
println("Hardest precision-recall area: ",
        round(area_under_curve(precision_recall); digits=4))
println("Mean cross-validation accuracy: ", round(mean_cv; digits=4))
println("Generated figures in: ", OUTPUT_DIRECTORY)
