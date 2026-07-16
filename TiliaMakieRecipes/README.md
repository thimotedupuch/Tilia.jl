# TiliaMakieRecipes

Makie plotting recipes for Tilia's semantic result types. This is a separate
package so that Tilia's core package has no plotting dependency or extension.

Load it alongside Tilia and Makie to enable plotting:

```julia
using Tilia, TiliaMakieRecipes, CairoMakie

result = confusion_matrix([:no, :yes], [:no, :yes])
plot(result)
```

The recipes provide diagnostic defaults while remaining fully customizable
with normal Makie keywords:

- confusion-matrix heatmaps include class-name ticks and cell annotations;
- ROC and calibration plots include chance/perfect-calibration reference lines;
- precision–recall, calibration, cross-validation, and optimization curves use
  meaningful labels and appropriate unit-square limits where applicable;
- calibration marker sizes reflect bin counts;
- permutation importance includes feature names and standard-deviation error bars;
- cross-validation plots include the mean and a one-standard-deviation band;
- optimization traces report convergence in the axis subtitle.

Reference lines and confusion-matrix values can be disabled with
`show_reference=false` and `show_values=false`. Styling attributes include
`color`, `linewidth`, `marker`, `markersize`, `colormap`, `referencecolor`,
`referencelinestyle`, and `errorcolor`. Axis defaults can be overridden through
Makie's `axis=(...)` keyword.

## Dimensionality reduction

Six composite recipes visualize fitted dimensionality-reduction models:

```julia
fitted = fit(PCA(n_components=4), X)

projectionplot(fitted, X; groups=labels)
screeplot(fitted; threshold=0.95)
biplot(fitted, X; groups=labels, feature_names=names)
loadingsplot(fitted; component=1, feature_names=names)
componentplot(fitted; shape=(28, 28), columns=4)
reconstructionplot(fitted, X; shape=(28, 28), observations=1:6)
```

Projection and biplot axes include PCA explained-variance percentages.
Projection plots mark labeled-group centroids. Scree plots combine individual
and cumulative variance with a configurable threshold. Component galleries
also support NMF, FastICA, truncated SVD, and random projection; sequential or
diverging color maps are selected from component sign semantics.

## Model diagnostics

Higher-level functions return complete Makie figures:

```julia
clusterplot(fitted_clusterer, X2)
dendrogram(fitted_hierarchy)
decisionboundaryplot(fitted_classifier, X2, y)
treeplot(fitted_tree; feature_names=names)
learningcurveplot(model, X, y)
validationcurveplot(model, X, y; parameter=:max_depth, values=1:8)
residualplot(fitted_regressor, X, y)
coefficientplot(fitted_linear_model; feature_names=names)
regularizationpathplot(Lasso(), X, y)
mixturedensityplot(fitted_gaussian_mixture, X2)
neighborhoodplot(fitted_nearest_neighbors, queries)
anomalyscoreplot(fitted_isolation_forest, X)
tuningheatmap(tuning_result; xparameter=:depth, yparameter=:rate)
tuningparallelplot(tuning_result)
modelcomparisonplot(cv_results; names=model_names)
```

Cluster plots support K-means, DBSCAN, Gaussian mixtures, and agglomerative
clustering, showing applicable decision regions, centers, covariance ellipses,
and noise observations. Regression diagnostics combine predicted-versus-actual,
residual, and residual-distribution panels. Tree nodes show split conditions,
sample counts, impurity, predictions, and class color. Tuning and comparison
plots consume Tilia's semantic `TuningResult` and `CrossValidationResult`
objects directly.

## Interactive 3D plots

Six `Axis3` visualizations work as static CairoMakie figures and become freely
rotatable when rendered with GLMakie:

```julia
projectionplot3d(fitted_projection, X; groups=labels)
clusterplot3d(fitted_clusterer, X3)
pointcloudplot(X3; groups=labels, edges=graph_edges)
neighborhoodplot3d(fitted_neighbors, queries)
regressionsurfaceplot(fitted_regressor, X2, y)
tuninglandscapeplot(tuning_result; xparameter=:depth, yparameter=:rate)
```

Cluster plots add centroids, noise markers, and translucent covariance
ellipsoids. Regression surfaces retain the observations and draw residual
stems to the fitted surface. Point clouds may use categorical groups or a
continuous scalar color channel.

## Partial dependence and probability simplex

The final explanatory plots cover nonlinear feature effects and three-class
predictive uncertainty:

```julia
partialdependenceplot(fitted, X; feature=3, ice=true)
partialdependenceplot(fitted, X; feature=(3,4), target=:virginica)
partialdependenceplot(fitted, X; feature=(3,4), target=:virginica, surface=true)
probabilitysimplexplot(fitted_classifier, X; groups=true_labels)
```

For classifiers, `target` selects the probability being averaged. One-feature
partial-dependence plots overlay sampled ICE curves and the empirical feature
distribution. Two-feature effects can be rendered as a static contour or an
interactive `Axis3` surface. The simplex accepts either a fitted classifier or
an `n × 3` probability matrix and supports coloring by observed or predicted
class.

For a local checkout, instantiate and run the isolated tests with:

```sh
julia --project=TiliaMakieRecipes/test -e 'using Pkg; Pkg.instantiate()'
julia --project=TiliaMakieRecipes/test TiliaMakieRecipes/test/runtests.jl
```
