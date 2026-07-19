# Models

Every model in Tilia is an immutable estimator specification implementing the
same lifecycle: declare, `fit`, then `predict`, `predict_proba`, `transform`, or
`partial_fit` according to declared capabilities. Constructor docstrings define
hyperparameters; [Model numerical contracts](model-semantics.md) documents
objectives, convergence, regularization, and conventions.

The tables below are a task-oriented map of the built-in estimators. They are
not a compatibility matrix: use `capabilities` or `model_catalog` when sparse
input, observation weights, probabilities, missing values, or incremental
updates are requirements.

## Choosing by task

### Regression

| Family | Estimators |
|:--|:--|
| Baseline and linear | `MeanRegressor`, `LinearRegression`, `RidgeRegression` |
| Sparse and incremental | `Lasso`, `ElasticNet`, `SGDRegressor` |
| Generalized linear | `PoissonRegression`, `GammaRegression`, `TweedieRegression` |
| Robust and distributional | `QuantileRegression`, `HuberRegression`, `TheilSenRegression`, `RANSACRegression` |
| Structured linear | `MARSRegressor`, `PartialLeastSquaresRegression` |
| Neighbors and kernels | `KNeighborsRegressor`, `KernelRidgeRegression`, `SupportVectorRegressor` |
| Trees and ensembles | `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor` |
| Shallow neural | `MLPRegressor` |

```julia
model = RidgeRegression(lambda=0.2)
fitted = fit(model, Xtrain, ytrain)
predictions = predict(fitted, Xtest)
root_mean_squared_error(ytest, predictions)
```

Poisson, Gamma, and Tweedie regression use penalized likelihood objectives and
support log or identity links. Their target domains follow the selected
Tweedie power: Gamma and powers above two require strictly positive targets;
Poisson and powers from one through two accept nonnegative targets. Robust
regressors provide smoothed quantile and Huber objectives, sampled Theil--Sen
estimation, and RANSAC consensus fitting. These estimators currently fit dense
numeric matrices on the CPU. GLMs, quantile, Huber, and RANSAC regression
accept observation weights; Theil--Sen regression does not.

### Classification

| Family | Estimators |
|:--|:--|
| Linear | `LogisticRegression`, `SparseLogisticRegression`, `SGDClassifier` |
| Ordered outcomes | `OrdinalRegression` |
| Probabilistic | `GaussianNaiveBayes`, `MultinomialNaiveBayes`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| Neighbors and kernels | `KNeighborsClassifier`, `SupportVectorClassifier` |
| Trees and ensembles | `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier` |
| Shallow neural | `MLPClassifier` |

```julia
classifier = fit(RandomForestClassifier(n_estimators=200), Xtrain, ytrain)
labels = predict(classifier, Xtest)
probabilities = predict_proba(classifier, Xtest)
confusion_matrix(ytest, labels)
```

Classification labels are ordered deterministically during fitting. The same
order controls probability columns, reports, schemas, and classification
diagnostics.

`OrdinalRegression` is a proportional-odds model. Consequently, the
deterministic order of its fitted `classes` is the outcome order used by the
cumulative link; inspect that field before interpreting its coefficients or
probability columns.

## Meta-estimators

Meta-estimators wrap or combine ordinary Tilia predictors while preserving the
same `fit`/`predict` lifecycle:

| Purpose | Estimators |
|:--|:--|
| Multiclass reduction | `OneVsRestClassifier`, `OneVsOneClassifier` |
| Multiple targets | `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain` |
| Resampling and aggregation | `BaggingClassifier`, `BaggingRegressor`, `VotingClassifier`, `VotingRegressor` |
| Learned combination | `StackingClassifier`, `StackingRegressor` |
| Target and decision adaptation | `TransformedTargetRegressor`, `CalibratedClassifier`, `ThresholdSelectionWrapper` |

```julia
voter = VotingClassifier(
    LogisticRegression(lambda=0.1),
    DecisionTreeClassifier(max_depth=4);
    voting=:soft,
)
fitted_voter = fit(voter, Xtrain, ytrain; context=FitContext(seed=42))
predict_proba(fitted_voter, Xtest)

robust = TransformedTargetRegressor(
    RidgeRegression(lambda=0.2);
    func=log1p,
    inverse_func=expm1,
)
```

Multi-output estimators and classifier chains expect a target matrix with one
output per column. Stacking and the calibration and threshold wrappers use
cross-validation to construct out-of-fold predictions; pass a seeded `KFold`
when shuffled folds are required. Soft voting, calibration, and threshold
selection require probabilistic base classifiers in practice. Threshold
selection is binary-only and optimizes `:f1`, `:accuracy`, or
`:balanced_accuracy` on a fixed grid from zero to one.

All meta-estimators currently declare dense CPU execution. For a constructed
meta-estimator, `capabilities(model).weights` is true only when every wrapped
estimator that receives observation weights supports them.

### Unsupervised learning and transformation

| Task | Estimators |
|:--|:--|
| Scaling and feature construction | `Standardize`, `MinMaxScale`, `RobustScale`, `Normalize`, `PolynomialFeatures`, `Impute`, `OneHotEncode` |
| Decomposition | `PCA`, `TruncatedSVD`, `NMF`, `RandomProjection`, `FastICA` |
| Clustering | `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `FeatureAgglomeration` |
| Density and anomaly detection | `GaussianMixture`, `IsolationForest` |
| Neighborhood and representation | `NearestNeighbors`, `BernoulliRBM` |

```julia
projection = fit(PCA(n_components=3), X)
scores = transform(projection, X)
reconstructed = inverse_transform(projection, scores)

mixture = fit(GaussianMixture(n_components=3), X)
components = predict(mixture, X)
responsibilities = predict_proba(mixture, X)
```

PCA centers its input; `TruncatedSVD` deliberately does not and is the usual
choice when centering would destroy sparse structure. Component and cluster
numbers are identifiers, not semantic class labels.

## Common operations

The operation is determined by the fitted estimator's task:

| Operation | Purpose |
|:--|:--|
| `fit(model, X, y)` | Fit a supervised predictor |
| `fit(model, X)` | Fit an unsupervised estimator or transformer |
| `predict(fitted, Xnew)` | Numeric, class, cluster, or anomaly prediction |
| `predict_proba(fitted, Xnew)` | Class probabilities or mixture responsibilities |
| `transform(fitted, Xnew)` | Learned feature representation |
| `inverse_transform(fitted, Z)` | Reconstruction where supported |
| `partial_fit(...)` | Incremental update where declared |
| `report(fitted)` | Structured fit and execution diagnostics |

Only `SGDClassifier` and `SGDRegressor` currently support `partial_fit`. It can
start from a specification or continue an existing fitted SGD model. Pass the
complete set with `classes=...` when a classifier's first batch does not contain
every class that later batches may contain.

Tilia raises an explicit error when an operation is not supported. Check first
when writing generic code:

```julia
declared = capabilities(model)
declared.task
declared.probabilistic
declared.partial_fit
```

## Finding a model programmatically

`model_catalog` filters registered estimator types using the same capability
declarations:

```julia
probabilistic_classifiers = model_catalog(
    task=:classification,
    probabilistic=true,
)

weighted_regressors = model_catalog(
    task=:regression,
    weights=true,
)
```

Each entry contains `type` and `capabilities`. Available filters are `task`,
`sparse`, `missing`, `weights`, `partial_fit`, and `probabilistic`.

## Context, determinism, and reports

Pass a `FitContext` to control backend, numerical policy, deterministic random
streams, and compilation cache:

```julia
context = FitContext(
    seed=42,
    numerics=NumericsPolicy(),
    backend=CPUBackend(),
)

fitted = fit(KMeans(n_clusters=4, n_init=10), X; context=context)
report(fitted)
```

Stochastic estimators derive named streams from the context, so scheduling
order does not redefine their random meaning. Reports record convergence,
iterations, objective history where applicable, warnings, backend behavior,
and deterministic stream metadata.

## Composition, evaluation, and visualization

Models compose with fitted preprocessing through `Chain` and the branching
operators documented in [Pipelines and graphs](pipelines-and-graphs.md).
Evaluate complete specifications with [Model selection](model-selection.md),
persist fitted state with [Persistence](persistence.md), and inspect results
through the separate [Makie visualization package](visualization.md).
