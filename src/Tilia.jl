"""
Tilia is a Julia-native classical machine-learning stack with immutable model
specifications, fitted state, semantic pipelines, and optional accelerators.
"""
module Tilia

using LinearAlgebra
using Random
using SparseArrays
using Statistics
using SHA
using TOML
using Tables
import Base: split

include("core/errors.jl")
include("core/reports.jl")
include("core/context.jl")
include("core/estimator.jl")
include("core/fitted.jl")
include("core/traits.jl")
include("core/derivatives.jl")
include("data/schema.jl")
include("data/categorical.jl")
include("data/column_table.jl")
include("data/table_adapter.jl")
include("data/dataset.jl")
include("kernels/Kernels.jl")
import .Kernels: log_loss, mean_squared_error, root_mean_squared_error
include("solvers/Solvers.jl")
include("graph/node.jl")
include("graph/graph.jl")
include("graph/builder.jl")
include("graph/validation.jl")
include("graph/interpreter.jl")
include("graph/execution_plan.jl")
include("graph/passes/dead_node_elimination.jl")
include("graph/passes/conversion_elimination.jl")
include("graph/passes/constant_folding.jl")
include("graph/passes/device_placement.jl")
include("preprocessing/standardize.jl")
include("preprocessing/feature_transforms.jl")
include("preprocessing/impute.jl")
include("preprocessing/encode.jl")
include("graph/composition.jl")
include("models/mean_regressor.jl")
include("models/linear/least_squares.jl")
include("models/linear/logistic.jl")
include("models/linear/sparse_regression.jl")
include("models/linear/sparse_logistic.jl")
include("models/linear/sgd.jl")
include("models/linear/mars.jl")
include("models/linear/partial_least_squares.jl")
include("models/decomposition/pca.jl")
include("models/decomposition/nmf.jl")
include("models/decomposition/random_projection.jl")
include("models/decomposition/fastica.jl")
include("models/clustering/kmeans.jl")
include("models/clustering/dbscan.jl")
include("models/clustering/agglomerative.jl")
include("models/clustering/feature_agglomeration.jl")
include("models/probabilistic/gaussian_classifiers.jl")
include("models/probabilistic/multinomial_naive_bayes.jl")
include("models/neighbors/nearest_neighbors.jl")
include("models/mixture/gaussian_mixture.jl")
include("models/trees/decision_tree.jl")
include("models/trees/forest.jl")
include("models/trees/hist_gradient_boosting.jl")
include("models/trees/isolation_forest.jl")
include("models/kernel/kernel_ridge.jl")
include("models/kernel/support_vector.jl")
include("models/neural/mlp.jl")
include("models/neural/rbm.jl")
include("core/catalog.jl")
include("graph/contracts.jl")
include("graph/passes/transform_fusion.jl")
include("graph/optimization.jl")
include("graph/tracing.jl")
include("metrics/regression.jl")
include("metrics/classification.jl")
include("inspection/permutation_importance.jl")
include("model_selection/split.jl")
include("model_selection/cross_validation.jl")
include("persistence/format.jl")
include("core/api.jl")
include("core/display.jl")
include("core/docstrings.jl")

export fit, predict, predict_proba, transform, inverse_transform, partial_fit
export evaluate, tune, report, save_model, load_model
export Chain, Parallel, ColumnMap, Select, Concatenate
export CPUBackend, ReactantBackend, NumericsPolicy, FitContext, CompilationCache
export default_context, derive_context
export ConfusionMatrix, ROCResult, PrecisionRecallResult, CalibrationResult
export PermutationImportanceResult
export CrossValidationResult, OptimizationTrace, TuningResult
export AbstractEstimator, AbstractFittedEstimator, AbstractTransformer, AbstractPredictor
export MeanRegressor, Standardize, Dataset, Schema, ColumnSchema
export MinMaxScale, RobustScale, Normalize, PolynomialFeatures
export Impute, OneHotEncode, ColumnTable, CategoricalColumn, column_table
export LinearRegression, RidgeRegression
export LogisticRegression
export Lasso, ElasticNet
export SparseLogisticRegression
export SGDClassifier, SGDRegressor
export MARSRegressor
export PartialLeastSquaresRegression
export PCA, TruncatedSVD, KMeans, DBSCAN, AgglomerativeClustering
export NMF
export RandomProjection
export FastICA
export FeatureAgglomeration
export GaussianNaiveBayes, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
export MultinomialNaiveBayes
export NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, kneighbors
export GaussianMixture
export DecisionTreeClassifier, DecisionTreeRegressor
export RandomForestClassifier, RandomForestRegressor
export ExtraTreesClassifier, ExtraTreesRegressor
export HistGradientBoostingClassifier, HistGradientBoostingRegressor
export IsolationForest, anomaly_score
export KernelRidgeRegression
export SupportVectorClassifier, SupportVectorRegressor
export MLPClassifier, MLPRegressor
export BernoulliRBM
export capabilities, input_contract, output_schema
export model_catalog
export accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
export roc_curve, precision_recall_curve, calibration_curve, area_under_curve
export permutation_importance
export log_loss, mean_squared_error, root_mean_squared_error
export train_test_split, KFold, split, cross_validate

end
