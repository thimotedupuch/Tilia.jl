using Random
using Statistics
using Tilia

const REPEATS = 3

function median_seconds(operation; warmup=true)
    warmup && operation()
    times = Float64[]
    for _ in 1:REPEATS
        GC.gc()
        push!(times, @elapsed operation())
    end
    median(times)
end

function result(name, seconds)
    println(name, '\t', round(seconds * 1_000; digits=3))
end

println("task\tmedian_ms")

rng = Xoshiro(20260715)
X = randn(rng, 20_000, 32)
y = X * randn(rng, 32) .+ 0.1 .* randn(rng, 20_000)
result("linear_regression_fit",
       median_seconds(() -> fit(LinearRegression(), X, y)))

rng = Xoshiro(20260715)
X = randn(rng, 10_000, 32)
result("pca_fit",
       median_seconds(() -> fit(PCA(n_components=8), X)))

rng = Xoshiro(20260715)
X = randn(rng, 5_000, 16)
result("kmeans_fit",
       median_seconds(() -> fit(KMeans(n_clusters=8, n_init=3,
                                        max_iterations=50), X)))

rng = Xoshiro(20260715)
X = randn(rng, 10_000, 16)
query = randn(rng, 200, 16)
neighbors = fit(NearestNeighbors(n_neighbors=5), X)
result("neighbors_query",
       median_seconds(() -> kneighbors(neighbors, query)))

rng = Xoshiro(20260715)
X = randn(rng, 1_000, 12)
y = ifelse.(X[:, 1] .+ 0.5 .* X[:, 2] .> 0, "positive", "negative")
result("decision_tree_fit",
       median_seconds(() -> fit(DecisionTreeClassifier(max_depth=6), X, y)))
