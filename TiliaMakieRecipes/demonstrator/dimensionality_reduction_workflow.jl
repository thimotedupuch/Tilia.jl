ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CairoMakie
using DataFrames
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__, "output", "dimensionality_reduction")
mkpath(OUTPUT_DIRECTORY)

iris = Iris(as_df=false)
X_iris = Matrix(transpose(iris.features))
y_iris = replace.(vec(iris.targets), "Iris-" => "")
iris_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
iris_pca = fit(PCA(n_components=4), X_iris)

projection = projectionplot(
    iris_pca, X_iris; groups=y_iris, title="Iris PCA projection",
    figure=(size=(900, 650),), markersize=12,
)
scree = screeplot(
    iris_pca; threshold=0.95, figure=(size=(900, 650),),
)
iris_biplot = biplot(
    iris_pca, X_iris; groups=y_iris, feature_names=iris_names,
    figure=(size=(900, 650),),
)
loadings = loadingsplot(
    iris_pca; component=1, feature_names=iris_names,
    figure=(size=(900, 650),),
)

mnist = MNIST(Float32, :train)
mnist_indices = reduce(vcat, findall(==(digit), mnist.targets)[1:150] for digit in 0:9)
X_mnist = permutedims(reshape(mnist.features[:, :, mnist_indices], :, length(mnist_indices)))
mnist_pca = fit(PCA(n_components=16), X_mnist)

components = componentplot(
    mnist_pca; shape=(28, 28), components=1:16, columns=4,
    title="MNIST principal components", figure=(size=(900, 900),),
)
reconstructions = reconstructionplot(
    mnist_pca, X_mnist; shape=(28, 28), observations=1:150:751,
    figure=(size=(1200, 650),), colormap=:grays,
)

plots = (
    ("iris_projection.png", projection),
    ("iris_scree.png", scree),
    ("iris_biplot.png", iris_biplot),
    ("iris_loadings.png", loadings),
    ("mnist_components.png", components),
    ("mnist_reconstruction.png", reconstructions),
)
for (filename, figure_axis_plot) in plots
    save(joinpath(OUTPUT_DIRECTORY, filename), figure_axis_plot.figure; px_per_unit=2)
end

dashboard = Figure(size=(1800, 1900), fontsize=17)
projectionplot(dashboard[1, 1], projection.plot.data[]; markersize=10)
screeplot(dashboard[1, 2], scree.plot.data[])
biplot(dashboard[2, 1], iris_biplot.plot.data[])
loadingsplot(dashboard[2, 2], loadings.plot.data[])
componentplot(dashboard[3, 1:2], components.plot.data[]; colormap=:balance)
reconstructionplot(dashboard[4, 1:2], reconstructions.plot.data[]; colormap=:grays)
Label(dashboard[0, 1:2], "Tilia dimensionality-reduction gallery",
      fontsize=32, font=:bold)
Label(dashboard[5, 1:2],
      "MLDatasets.Iris · MLDatasets.MNIST · PCA fitted with Tilia",
      fontsize=18)
save(joinpath(OUTPUT_DIRECTORY, "dimensionality_reduction_dashboard.png"),
     dashboard; px_per_unit=1.5)

println("Iris first two components explain ",
        round(100sum(iris_pca.explained_variance_ratio[1:2]); digits=1), "% of variance")
println("MNIST reconstruction uses 16 components for 784 pixels")
println("Generated figures in: ", OUTPUT_DIRECTORY)
