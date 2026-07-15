# Model numerical contracts

All estimators accept `Float32` and `Float64` in the conformance matrix and
preserve the input floating type for numerical predictions or transforms.
Classification labels are sorted once during fitting and that order is stored
in the fitted schema and report. Missing or non-finite numerical input is an
error unless an explicit preprocessing or numerical policy says otherwise.

| Estimator | Objective and method | Stopping, regularization, and conventions |
|:--|:--|:--|
| `MeanRegressor` | Arithmetic or frequency-weighted target mean; stable policy-controlled accumulation. | No iteration or intercept; population convention. |
| `LinearRegression` | Minimize weighted squared residuals by pivoted QR or minimum-norm SVD. | Unpenalized intercept is recovered after centering; numerical-rank tolerance comes from `NumericsPolicy`. |
| `RidgeRegression` | Squared residuals plus `lambda*sum(abs2, coefficients)`, solved by Cholesky or SVD. | Intercept is unpenalized; rank tolerance comes from `NumericsPolicy`. |
| `LogisticRegression` | Weighted binary logistic loss plus `lambda/2*sum(abs2, coefficients)`; analytic damped Newton, one-vs-rest for multiclass. | Gradient-norm tolerance or `max_iterations`; unpenalized intercept; sorted class order. |
| `Lasso` | Squared loss plus an L1 penalty; cyclic coordinate descent. | Maximum-coordinate-update tolerance or `max_iterations`; centered, unpenalized intercept. |
| `ElasticNet` | Squared loss plus `lambda*l1_ratio` L1 and `lambda*(1-l1_ratio)` L2 penalties; coordinate descent. | Same convergence and intercept convention as Lasso. |
| `SparseLogisticRegression` | Weighted logistic loss with L1/L2 elastic-net terms; proximal gradient. | Parameter-change tolerance or `max_iterations`; unpenalized intercept and sorted classes. |
| `Standardize` | Per-feature population mean and standard deviation using the policy accumulation type. | Zero-variance scales become one; sparse centering errors or explicitly densifies according to `NumericsPolicy`. |
| `MinMaxScale` | Per-feature affine mapping from training extrema into `feature_range`. | Constant features map to the lower endpoint; optional clipping applies only during transformation. |
| `RobustScale` | Per-feature median centering and interquantile scaling. | Zero quantile ranges use a scale of one; centering and scaling are independently optional. |
| `Normalize` | Per-observation L1, L2, or maximum-norm scaling. | Zero rows remain zero; sparse inputs preserve sparse structure. |
| `PolynomialFeatures` | Deterministic monomials ordered by total degree and feature index. | Degree is limited to 32 and output width to 100,000 columns, rejecting accidental combinatorial expansion before allocation. |
| `PCA` | SVD of centered observations; leading right singular vectors. | Explained variance uses the documented sample covariance convention; component signs are canonicalized. |
| `TruncatedSVD` | Leading singular vectors without centering, preserving sparse semantics. | Fixed component count; singular values are nonnegative and ordered. |
| `KMeans` | Minimize within-cluster squared Euclidean distance by Lloyd updates with deterministic named restarts. | Center displacement tolerance or `max_iterations`; the lowest-inertia restart wins. |
| `GaussianNaiveBayes` | Maximum-likelihood class priors, means, and diagonal Gaussian variances. | Population weighted variance plus variance smoothing; sorted classes. |
| `LinearDiscriminantAnalysis` | Gaussian discriminants with a pooled covariance estimate. | Eigenvalue regularization controls singular covariance; sorted classes. |
| `QuadraticDiscriminantAnalysis` | Class-specific Gaussian covariance discriminants. | Covariance regularization is applied per class; sorted classes. |
| `GaussianMixture` | Maximize Gaussian-mixture log likelihood by expectation-maximization with named restarts. | Log-likelihood-change tolerance or `max_iterations`; covariance regularization prevents singular components. |
| `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` | Exact brute-force ranking by the selected metric. | Stable index tie-breaking; classifier votes follow stored class order; regression uses the arithmetic neighbor mean. |
| `DecisionTreeClassifier` | Greedy CART impurity decrease using Gini or entropy. | Depth, leaf-size, and impurity constraints stop growth; ties follow sorted classes and feature/split order. |
| `DecisionTreeRegressor` | Greedy CART squared-error reduction. | Depth, leaf-size, and impurity constraints stop growth; leaves predict their target mean. |
| `RandomForestClassifier`, `RandomForestRegressor` | Average independently bootstrapped randomized CART trees. | Named per-tree streams make scheduling order irrelevant; classifier probabilities follow sorted classes. |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | Average trees using randomized split candidates. | Same stopping, stream, class, and aggregation conventions as forests. |
| `HistGradientBoostingClassifier` | Stagewise logistic residual fitting with quantile-binned regression trees. | `n_estimators` is the stage limit; learning rate shrinks updates; sorted binary classes. |
| `HistGradientBoostingRegressor` | Stagewise squared-error residual fitting with quantile-binned trees. | `n_estimators` is the stage limit; initial prediction is the target mean. |
| `IsolationForest` | Random recursive isolation; anomaly score derives from mean path length. | Tree depth and sampled observation count bound construction; named tree streams are deterministic. |
| `KernelRidgeRegression` | Squared loss in the kernel dual with diagonal ridge regularization. | Direct linear solve; intercept is represented by the chosen kernel formulation. |
| `SupportVectorClassifier` | Squared hinge loss plus RKHS norm in kernel coefficients; batch gradient descent. | Maximum-parameter-update tolerance or `max_iterations`; explicit unpenalized intercept and sorted classes. |
| `SupportVectorRegressor` | Squared epsilon-insensitive loss plus RKHS norm; batch gradient descent. | Maximum-parameter-update tolerance or `max_iterations`; explicit unpenalized intercept. |
| `MLPClassifier`, `MLPRegressor` | One-hidden-layer batch backpropagation; cross-entropy for classification and squared error for regression. | Fixed iteration limit and learning rate; biases are unregularized; classifier outputs use sorted classes. |
| `BernoulliRBM` | Bernoulli reconstruction likelihood approximation by contrastive divergence. | Fixed epochs and batches; named initialization/epoch streams; visible and hidden biases are explicit. |

Fit reports record the concrete solver status, objective history where
applicable, iteration count, numerical policy, regularization, backend,
deterministic stream, and warnings. Constructor-specific defaults and exact
parameter meanings are in the corresponding API docstrings.
