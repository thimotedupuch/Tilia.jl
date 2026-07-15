# Metrics

Regression metrics include mean squared error and RMSE. Classification metrics
include accuracy, precision, recall, F1, log loss, confusion matrices, binary
ROC and precision–recall curves, and reliability calibration data.

```julia
root_mean_squared_error(y, predict(fitted, X))
accuracy_score(labels, predict(classifier, X))
confusion_matrix(labels, predict(classifier, X))
roc = roc_curve(labels, predict_proba(classifier, X)[:, 2])
area_under_curve(roc)
calibration_curve(labels, predict_proba(classifier, X)[:, 2])
```

Class ordering is explicit in result objects and fitted schemas.

Model-agnostic inspection uses deterministic named random streams:

```julia
importance = permutation_importance(fitted, X, y;
                                    n_repeats=5,
                                    context=FitContext(seed=42))
importance.mean_importance
```
