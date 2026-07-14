# Getting started

Observations are rows and features are columns. Fit returns new state and never
changes the model specification.

```julia
using Tilia
X = [1.0 10; 2 20; 3 30; 4 40]
y = [2.0, 4, 6, 8]
fitted = fit(Chain(Standardize(), RidgeRegression(lambda=0.1)), X, y)
predictions = predict(fitted, X)
diagnostics = report(fitted)
```
