# Metrics and inspection

Tilia metrics return ordinary numbers or semantic result objects that retain
the coordinates and labels needed for reporting and visualization. Inputs are
validated for equal lengths, finite scores, valid probabilities, and explicit
class order.

## Regression metrics

```julia
predictions = predict(fitted_regressor, Xtest)

mean_squared_error(ytest, predictions)
root_mean_squared_error(ytest, predictions)
```

These public functions use the same centralized kernels as Tilia's model and
benchmark code, preventing metric semantics from diverging between internal
and user-facing paths.

## Classification labels

```julia
predictions = predict(fitted_classifier, Xtest)

accuracy_score(ytest, predictions)
precision_score(ytest, predictions; average=:macro)
recall_score(ytest, predictions; average=:weighted)
f1_score(ytest, predictions; average=:none)
```

Precision, recall, and F1 support `:none`, `:macro`, `:micro`, and `:weighted`
averaging. `zero_division` controls undefined class ratios. Optional
nonnegative observation weights are supported.

Confusion matrices keep actual classes in rows and predicted classes in
columns:

```julia
matrix = confusion_matrix(ytest, predictions; labels=fitted_classifier.classes)
matrix.matrix
matrix.labels
```

Supplying `labels` is useful when a test subset does not contain every fitted
class. Duplicate labels or values outside the requested order are rejected.

## Probabilistic classification

Probability columns must match an explicit label order:

```julia
probabilities = predict_proba(fitted_classifier, Xtest)
classes = fitted_classifier.classes

log_loss(ytest, probabilities; labels=classes)
```

For a binary classifier, select the probability column for the intended
positive class rather than assuming it is always column two:

```julia
positive = :yes
column = findfirst(==(positive), classes)
scores = probabilities[:, column]

roc = roc_curve(ytest, scores; positive_label=positive)
precision_recall = precision_recall_curve(ytest, scores; positive_label=positive)

area_under_curve(roc)
area_under_curve(precision_recall)
```

`ROCResult` stores false-positive rate, true-positive rate, and thresholds.
`PrecisionRecallResult` stores precision, recall, and thresholds. Tied scores
are processed as groups with deterministic index ordering.

## Calibration

Reliability data compares predicted probability with observed positive
frequency:

```julia
calibration = calibration_curve(
    ytest, scores;
    positive_label=positive,
    n_bins=10,
    strategy=:quantile,
)
```

`strategy=:uniform` uses equally spaced probability bins;
`strategy=:quantile` derives bins from the score distribution. Empty bins are
omitted from the returned coordinates, while bin edges and weighted counts
remain available.

## Permutation importance

Permutation importance repeatedly disrupts one feature and measures the score
decrease:

```julia
importance = permutation_importance(
    fitted, Xtest, ytest;
    n_repeats=10,
    context=FitContext(seed=42),
)

importance.baseline_score
importance.mean_importance
importance.standard_deviation
importance.feature_names
```

The default score is accuracy for classification and negative RMSE for
regression, so positive importance consistently means that permutation harmed
predictive quality. A custom `scoring(truth, prediction)` function can be
provided; use `greater_is_better=false` when smaller custom scores are better.

Permutation streams are derived by feature and repetition from the supplied
context, making results reproducible without depending on global RNG state.

## Plotting semantic results

`ConfusionMatrix`, curve, calibration, permutation-importance, and
cross-validation results are understood by the separate
`TiliaMakieRecipes` package. See [Visualization with Makie](visualization.md)
for `plot(result)` recipes and composite diagnostic figures.
