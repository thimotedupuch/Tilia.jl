const REGISTERED_ESTIMATOR_TYPES = (
    MeanRegressor, Standardize, MinMaxScale, RobustScale, Normalize, PolynomialFeatures,
    Impute, OneHotEncode,
    LinearRegression, RidgeRegression, LogisticRegression,
    Lasso, ElasticNet, SparseLogisticRegression,
    SGDClassifier, SGDRegressor,
    MARSRegressor,
    PartialLeastSquaresRegression,
    PCA, TruncatedSVD, NMF, RandomProjection, FastICA,
    KMeans, DBSCAN, AgglomerativeClustering, FeatureAgglomeration,
    GaussianNaiveBayes, MultinomialNaiveBayes,
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
    GaussianMixture,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    IsolationForest, KernelRidgeRegression,
    SupportVectorClassifier, SupportVectorRegressor,
    MLPClassifier, MLPRegressor, BernoulliRBM,
    Select, Parallel, ColumnMap, Concatenate,
)

"""Discover registered estimator types using their declared capabilities.

Each non-`nothing` keyword restricts the returned entries. Entries contain the
estimator `type` and its machine-readable `capabilities` tuple.

# Example

```julia
classifiers = model_catalog(task=:classification, probabilistic=true)
all(entry -> entry.capabilities.task == :classification, classifiers)
```
"""
function model_catalog(; task=nothing, sparse=nothing, missing=nothing,
                       weights=nothing, partial_fit=nothing, probabilistic=nothing)
    filters = (; task, sparse, missing, weights, partial_fit, probabilistic)
    entries = NamedTuple[]
    for estimator_type in REGISTERED_ESTIMATOR_TYPES
        declared = capabilities(estimator_type)
        matches = all(pairs(filters)) do (name, expected)
            expected === nothing || getproperty(declared, name) == expected
        end
        matches && push!(entries, (type=estimator_type, capabilities=declared))
    end
    entries
end
