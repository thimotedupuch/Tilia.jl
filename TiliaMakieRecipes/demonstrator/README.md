# TiliaMakieRecipes demonstrator

This isolated environment renders example PNGs for every diagnostic recipe and
a combined dashboard using CairoMakie.

```sh
julia TiliaMakieRecipes/demonstrator/setup.jl
julia --project=TiliaMakieRecipes/demonstrator \
  TiliaMakieRecipes/demonstrator/generate_figures.jl
```

Generated files are written to `output/`. Both scripts set
`ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1` before loading or modifying packages.

## Real Wine classification workflow

`wine_workflow.jl` loads `MLDatasets.Wine`, performs a deterministic stratified
train/test split, standardizes its 13 chemical measurements, fits Tilia's
multiclass logistic regression, and computes diagnostics from held-out
predictions. It also performs eight-fold cross-validation and permutation
importance.

```sh
julia --project=TiliaMakieRecipes/demonstrator \
  TiliaMakieRecipes/demonstrator/wine_workflow.jl
```

The seven individual diagnostic plots and a combined workflow dashboard are
written to `output/wine_workflow/`.

## Larger MNIST workflow

`mnist_workflow.jl` uses balanced subsets of `MLDatasets.MNIST`: 5,000 training
and 2,000 official-test images across ten digit classes. It reduces the 784
pixels to 35 PCA components, fits multiclass logistic regression, and chooses
the hardest held-out digit for one-vs-rest ROC, precision–recall, and
calibration diagnostics. Five-fold validation uses 3,000 observations and
refits PCA inside every fold to prevent leakage.

```sh
julia --project=TiliaMakieRecipes/demonstrator \
  TiliaMakieRecipes/demonstrator/mnist_workflow.jl
```

The resulting 10×10 confusion matrix, individual diagnostics, and dashboard
are written to `output/mnist_workflow/`. On first use, MLDatasets downloads the
official MNIST files.

## Dimensionality-reduction gallery

`dimensionality_reduction_workflow.jl` creates all six dimensionality-reduction
figures. Iris supplies labeled PCA projections, a scree plot, biplot, and
loadings. MNIST supplies a 4×4 principal-component gallery and original versus
reconstructed digit panels.

```sh
julia --project=TiliaMakieRecipes/demonstrator \
  TiliaMakieRecipes/demonstrator/dimensionality_reduction_workflow.jl
```

Outputs are written to `output/dimensionality_reduction/`.
