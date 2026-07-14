# Model selection

Splits are deterministic and never overlap. Cross-validation refits the entire
pipeline inside every fold.

```julia
cv = KFold(5; shuffle=true, seed=42)
evaluation = evaluate(model, X, y; cv=cv)
tuned = tune(RidgeRegression(), X, y;
             parameter_grid=(lambda=[0.01, 0.1, 1.0],), cv=cv)
```

Classification scores are maximized and regression scores minimized by
default. Specify `maximize` when supplying a custom score with different
semantics.
