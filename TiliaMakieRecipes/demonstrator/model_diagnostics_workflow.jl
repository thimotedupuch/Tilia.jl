ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CairoMakie
using DataFrames
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output", "model_diagnostics")
mkpath(OUTPUT_DIRECTORY)

iris = Iris(as_df=false)
X = Float64.(Matrix(transpose(iris.features)))
y = replace.(vec(iris.targets), "Iris-" => "")
X2 = X[:, 3:4]
feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

kmeans = fit(KMeans(n_clusters=3, n_init=5), X2)
dbscan = fit(DBSCAN(radius=0.34, min_neighbors=4), X2)
mixture = fit(GaussianMixture(n_components=3, n_init=3), X2)
agglomerative = fit(AgglomerativeClustering(n_clusters=3), X2)
full_hierarchy = fit(AgglomerativeClustering(n_clusters=1), X2)
feature_hierarchy = fit(FeatureAgglomeration(n_clusters=1), X)
tree = fit(DecisionTreeClassifier(max_depth=3), X2, y)
linear = fit(LinearRegression(), X[:, 1:3], X[:, 4])
neighbors = fit(NearestNeighbors(n_neighbors=5), X2)
anomaly_X = vcat(X2, [0.4 2.8; 8.2 0.2; 7.8 3.1; 1.0 -0.3])
forest = fit(IsolationForest(n_estimators=80, contamination=0.05), anomaly_X)

tuning = tune(DecisionTreeClassifier(), X, y;
    parameter_grid=(max_depth=[1, 2, 3, 5], min_samples_leaf=[1, 3, 6]),
    cv=KFold(4; shuffle=true, seed=7))
comparisons = [
    cross_validate(DecisionTreeClassifier(max_depth=3), X, y; cv=KFold(5; shuffle=true, seed=12)),
    cross_validate(KNeighborsClassifier(n_neighbors=7), X, y; cv=KFold(5; shuffle=true, seed=12)),
    cross_validate(LogisticRegression(max_iterations=300), X, y; cv=KFold(5; shuffle=true, seed=12)),
]

figures = (
    ("cluster_kmeans.png", clusterplot(kmeans, X2; title="K-means · Iris petals")),
    ("cluster_dbscan.png", clusterplot(dbscan, X2; decision_regions=false, title="DBSCAN · noise and dense groups")),
    ("cluster_gaussian_mixture.png", clusterplot(mixture, X2; title="Gaussian mixture · covariance ellipses")),
    ("cluster_agglomerative.png", clusterplot(agglomerative, X2; title="Agglomerative clusters")),
    ("dendrogram_samples.png", TiliaMakieRecipes.dendrogram(full_hierarchy; labels=1:size(X,1), title="Iris sample hierarchy")),
    ("dendrogram_features.png", TiliaMakieRecipes.dendrogram(feature_hierarchy; labels=feature_names, title="Feature agglomeration")),
    ("decision_boundary.png", decisionboundaryplot(tree, X2, y; title="Decision-tree regions · Iris petals")),
    ("decision_tree.png", treeplot(tree; feature_names=["Petal length", "Petal width"])),
    ("learning_curve.png", learningcurveplot(DecisionTreeClassifier(max_depth=3), X, y)),
    ("validation_curve.png", validationcurveplot(DecisionTreeClassifier(), X, y; parameter=:max_depth, values=[1,2,3,4,6,10])),
    ("regression_diagnostics.png", residualplot(linear, X[:,1:3], X[:,4]; title="Predicting Iris petal width")),
    ("coefficient_plot.png", coefficientplot(linear; feature_names=feature_names[1:3])),
    ("regularization_path.png", regularizationpathplot(ElasticNet(l1_ratio=0.65, max_iterations=500), X[:,1:3], X[:,4]; feature_names=feature_names[1:3])),
    ("gaussian_mixture_density.png", mixturedensityplot(mixture, X2)),
    ("nearest_neighbors.png", neighborhoodplot(neighbors, X2[[1, 55, 125], :]; n_neighbors=5)),
    ("isolation_forest_scores.png", anomalyscoreplot(forest, anomaly_X)),
    ("tuning_heatmap.png", tuningheatmap(tuning; xparameter=:max_depth, yparameter=:min_samples_leaf)),
    ("tuning_parallel_coordinates.png", tuningparallelplot(tuning)),
    ("model_comparison.png", modelcomparisonplot(comparisons; names=["Tree", "7-NN", "Logistic"])),
)

for (filename, figure) in figures
    save(joinpath(OUTPUT_DIRECTORY, filename), figure; px_per_unit=1.7)
end

println("Generated ", length(figures), " figures in: ", OUTPUT_DIRECTORY)
