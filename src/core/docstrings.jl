@doc """Fit an immutable estimator specification and return fitted state.

```julia
fitted = fit(MeanRegressor(), [1.0;; 2.0], [1.0, 3.0])
```
""" fit

@doc """Predict outputs from fitted state. `predict(fitted, X)` keeps observations in rows.""" predict
@doc """Return class or component probabilities from supported fitted estimators.""" predict_proba
@doc """Apply a fitted transformer to new observations.""" transform
@doc """Map transformed observations back to feature space when supported.""" inverse_transform
@doc """Update estimators that explicitly declare online-learning support; unsupported models raise `MethodError`.""" partial_fit
@doc """Return the machine-readable `FitReport` associated with fitted state.""" report
@doc """Describe accepted input semantics for an estimator.""" input_contract
@doc """Describe the output schema rule for an estimator.""" output_schema
@doc """Create a fresh deterministic CPU fit context.""" default_context

@doc """Abstract supertype for feature transformers such as `Standardize`.""" AbstractTransformer
@doc """Abstract supertype for predictors such as `LinearRegression`.""" AbstractPredictor
@doc """Host CPU execution backend. Use `FitContext(backend=CPUBackend())`.""" CPUBackend
@doc """Optional Reactant backend with explicit device and fallback policy.""" ReactantBackend

@doc """Select matrix indices or table column names, for example `Select(:age, :income)`.""" Select
@doc """Fit several transformers to the same input, for example `Parallel(PCA(), Standardize())`.""" Parallel
@doc """Concatenate tuple-valued transformer branches column-wise.""" Concatenate
@doc """Apply transformers to named column groups, for example `ColumnMap(:age => Standardize())`.""" ColumnMap

@doc """Center and/or scale numeric features with population standard deviations.""" Standardize
@doc """Exact brute-force neighbor index. Call `kneighbors(fit(NearestNeighbors(), X), query)`.""" NearestNeighbors

@doc """Numeric CART classifier with Gini or entropy impurity.""" DecisionTreeClassifier
@doc """Numeric CART regressor with squared-error impurity.""" DecisionTreeRegressor
@doc """Bootstrap ensemble of randomized classification trees.""" RandomForestClassifier
@doc """Bootstrap ensemble of randomized regression trees.""" RandomForestRegressor
@doc """Extremely randomized classification-tree ensemble.""" ExtraTreesClassifier
@doc """Extremely randomized regression-tree ensemble.""" ExtraTreesRegressor
@doc """Quantile-binned gradient-boosting classifier.""" HistGradientBoostingClassifier
@doc """Quantile-binned gradient-boosting regressor.""" HistGradientBoostingRegressor
@doc """One-hidden-layer probabilistic classifier trained by batch backpropagation.""" MLPClassifier
@doc """One-hidden-layer regressor trained by batch backpropagation.""" MLPRegressor

@doc """Confusion-matrix values and deterministic label ordering.""" ConfusionMatrix
@doc """False-positive rates, true-positive rates, and thresholds for an ROC curve.""" ROCResult
@doc """Fold scores, fitted folds, reports, and exact split indices.""" CrossValidationResult
@doc """Objective values and convergence status from iterative optimization.""" OptimizationTrace

@doc """Mean squared prediction error, optionally weighted.""" mean_squared_error
@doc """Square root of `mean_squared_error`, optionally weighted.""" root_mean_squared_error
