using Tilia
using Random

rng = Xoshiro(42)

function timed(operation, label)
    first_call = @elapsed result = operation()
    steady_state = @elapsed operation()
    println((benchmark=label, first_call_seconds=first_call,
             steady_state_seconds=steady_state))
    result
end

for observations in (100, 1_000, 10_000)
    features = 16
    X = randn(rng, observations, features)
    timed((model=:pca, observations=observations, features=features)) do
        fit(PCA(n_components=8), X)
    end
    fitted_neighbors = fit(NearestNeighbors(n_neighbors=5), X)
    query = view(X, 1:min(observations, 100), :)
    timed((model=:nearest_neighbors, observations=observations, features=features)) do
        kneighbors(fitted_neighbors, query)
    end
end

for observations in (100, 1_000)
    X = vcat(randn(rng, observations ÷ 2, 8) .- 2,
             randn(rng, observations - observations ÷ 2, 8) .+ 2)
    timed((model=:kmeans, observations=observations, features=8)) do
        fit(KMeans(n_clusters=2, n_init=3), X)
    end
    timed((model=:gaussian_mixture, observations=observations, features=8)) do
        fit(GaussianMixture(n_components=2, n_init=2), X)
    end
end
