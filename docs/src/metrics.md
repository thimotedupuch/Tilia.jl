# Metrics

Regression metrics include mean squared error and RMSE. Classification metrics
include accuracy, precision, recall, F1, log loss, and confusion matrices.

```julia
root_mean_squared_error(y, predict(fitted, X))
accuracy_score(labels, predict(classifier, X))
confusion_matrix(labels, predict(classifier, X))
```

Class ordering is explicit in result objects and fitted schemas.
