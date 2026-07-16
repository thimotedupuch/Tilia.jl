ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CairoMakie
using DataFrames
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output", "three_dimensional")
mkpath(OUTPUT_DIRECTORY)

iris = Iris(as_df=false)
X = Float64.(Matrix(transpose(iris.features)))
y = replace.(vec(iris.targets), "Iris-" => "")
X3 = X[:, 1:3]

pca = fit(PCA(n_components=4), X)
mixture = fit(GaussianMixture(n_components=3, n_init=3), X3)
neighbors = fit(NearestNeighbors(n_neighbors=6), X3)
regression_X = X[:, [1,3]]
regressor = fit(RidgeRegression(lambda=0.15), regression_X, X[:,4])
tuning = tune(DecisionTreeClassifier(), X, y;
    parameter_grid=(max_depth=[1,2,3,5,8], min_samples_leaf=[1,2,4,8]),
    cv=KFold(4;shuffle=true,seed=29))

edges = [(i, i+1) for i in 1:149 if y[i] == y[i+1]]
figures = (
    ("iris_pca_projection_3d.png", projectionplot3d(pca,X;groups=y,title="Iris · first three principal components")),
    ("iris_gaussian_mixture_3d.png", clusterplot3d(mixture,X3;title="Iris · Gaussian-mixture ellipsoids")),
    ("iris_point_cloud_3d.png", pointcloudplot(X3;groups=y,edges=edges,labels=("Sepal length","Sepal width","Petal length"),title="Iris feature-space point cloud")),
    ("iris_neighbors_3d.png", neighborhoodplot3d(neighbors,X3[[15,75,135],:];n_neighbors=6,title="Iris nearest-neighbor neighborhoods")),
    ("iris_regression_surface_3d.png", regressionsurfaceplot(regressor,regression_X,X[:,4];title="Petal width response surface")),
    ("tree_tuning_landscape_3d.png", tuninglandscapeplot(tuning;xparameter=:max_depth,yparameter=:min_samples_leaf,title="Decision-tree CV landscape")),
)

for (filename,figure) in figures
    save(joinpath(OUTPUT_DIRECTORY,filename),figure;px_per_unit=1.8)
end

println("Generated ",length(figures)," Axis3 figures in: ",OUTPUT_DIRECTORY)
