# Visualization with Makie

Plotting is provided by the separate `TiliaMakieRecipes` package. Tilia itself
does not depend on Makie and contains no plotting extension, so applications
that do not visualize results pay no plotting dependency or compilation cost.

Choose a Makie backend in the application. CairoMakie is convenient for PNG,
SVG, and PDF output, while GLMakie provides interactive windows and rotatable
three-dimensional figures.

```julia
using Tilia
using TiliaMakieRecipes
using CairoMakie # or GLMakie
```

For a checkout of this repository, the plotting package and its demonstrator
environment are located in `TiliaMakieRecipes/`. The demonstrator can be
instantiated independently:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=TiliaMakieRecipes/demonstrator \
    -e 'using Pkg; Pkg.instantiate()'
```

## Semantic result recipes

Loading `TiliaMakieRecipes` enables ordinary `plot(result)` calls for Tilia's
semantic report types:

```julia
evaluation = cross_validate(model, X, y; cv=KFold(5))
figure_axis_plot = plot(evaluation)
save("cross_validation.png", figure_axis_plot)
```

| Tilia result | Default visualization |
|:--|:--|
| `ConfusionMatrix` | Annotated class heatmap |
| `ROCResult` | ROC curve and chance reference |
| `PrecisionRecallResult` | Precision--recall curve |
| `CalibrationResult` | Reliability curve and perfect-calibration reference |
| `PermutationImportanceResult` | Feature bars with uncertainty |
| `CrossValidationResult` | Integer fold axis, mean, and standard-deviation band |
| `OptimizationTrace` | Objective history and convergence status |

Normal Makie keywords remain available. Common recipe attributes include
`color`, `colormap`, `linewidth`, `marker`, `markersize`, `show_reference`, and
`show_values`; axis settings can be overridden with `axis=(...)`.

## Plotting API

The higher-level functions return complete Makie figures unless noted
otherwise.

### Evaluation and model inspection

| Function | Purpose |
|:--|:--|
| `decisionboundaryplot(fitted, X, y)` | Decision regions for any two-feature classifier |
| `treeplot(fitted; feature_names)` | Splits, samples, impurity, and class-colored predictions |
| `learningcurveplot(model, X, y)` | Training and validation score versus training-set size |
| `validationcurveplot(model, X, y; ...)` | Training and validation score versus one hyperparameter |
| `residualplot(fitted, X, y)` | Predicted-versus-actual, residual, and residual-distribution panels |
| `coefficientplot(fitted)` | Linear, logistic, sparse, and PLS coefficients |
| `regularizationpathplot(model, X, y)` | Lasso or elastic-net coefficient paths |
| `anomalyscoreplot(fitted, X)` | Isolation-forest scores and fitted threshold |

### Clustering, mixtures, and neighbors

| Function | Purpose |
|:--|:--|
| `clusterplot(fitted, X)` | K-means, DBSCAN, Gaussian-mixture, or agglomerative assignments with applicable regions, centers, ellipses, and noise |
| `dendrogram(fitted)` | Observation or feature-agglomeration merge tree |
| `mixturedensityplot(fitted, X)` | Gaussian-mixture density contours |
| `neighborhoodplot(fitted, queries)` | Queries connected to their nearest neighbors |

### Dimensionality reduction

```julia
fitted = fit(PCA(n_components=4), X)

projectionplot(fitted, X; groups=labels)
screeplot(fitted; threshold=0.95)
biplot(fitted, X; groups=labels, feature_names=names)
loadingsplot(fitted; component=1, feature_names=names)
componentplot(fitted; shape=(28, 28))
reconstructionplot(fitted, X; shape=(28, 28))
```

Component galleries also accept supported fitted NMF, FastICA, truncated-SVD,
and random-projection models.

### Tuning and comparison

| Function | Purpose |
|:--|:--|
| `tuningheatmap(result; xparameter, yparameter)` | Two-parameter tuning grid |
| `tuningparallelplot(result)` | All trials as parallel coordinates |
| `modelcomparisonplot(results; names)` | Fold-score distributions for multiple cross-validation results |

### Explanation

| Function | Purpose |
|:--|:--|
| `partialdependenceplot(fitted, X; feature=i)` | Partial dependence, sampled ICE curves, and empirical feature distribution |
| `partialdependenceplot(fitted, X; feature=(i,j))` | Two-feature contour, or an `Axis3` surface with `surface=true` |
| `probabilitysimplexplot(fitted, X; groups=y)` | Ternary uncertainty plot for exactly three classes; also accepts an `n × 3` probability matrix |

## Three-dimensional API

These functions build `Makie.Axis3` figures. They render statically with
CairoMakie and remain rotatable with GLMakie:

```julia
projectionplot3d(fitted_projection, X; groups=labels)
clusterplot3d(fitted_clusterer, X3)
pointcloudplot(X3; groups=labels, edges=edges)
neighborhoodplot3d(fitted_neighbors, queries)
regressionsurfaceplot(fitted_regressor, X2, y)
tuninglandscapeplot(tuning_result; xparameter=:depth, yparameter=:rate)
```

Three-dimensional cluster plots include centroids, noise points, and
translucent covariance ellipsoids. Regression surfaces retain observations and
draw residual stems to the fitted surface.

## Examples and generated galleries

Reproducible CairoMakie workflows are available under
`TiliaMakieRecipes/demonstrator/`. They cover synthetic diagnostics, real
MLDatasets workflows, dimensionality reduction, model diagnostics, `Axis3`
figures, partial dependence, and probability simplexes. Generated PNG files
are written below `TiliaMakieRecipes/demonstrator/output/`.
