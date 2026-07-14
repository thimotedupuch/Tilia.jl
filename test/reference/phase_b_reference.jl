@testset "Phase B scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "phase_b_sklearn.toml"))
    @test fixture["source"]["package"] == "scikit-learn"
    @test fixture["source"]["version"] == "1.9.0"
    atol = fixture["tolerance"]["absolute"]
    rtol = fixture["tolerance"]["relative"]
    X = reduce(vcat, permutedims.(fixture["input"]["X"]))
    query = reduce(vcat, permutedims.(fixture["input"]["query"]))
    y = fixture["input"]["y"]

    pca = fit(PCA(n_components=2), X)
    @test pca.explained_variance ≈ fixture["pca"]["explained_variance"] atol=atol rtol=rtol
    @test pca.explained_variance_ratio ≈ fixture["pca"]["explained_variance_ratio"] atol=atol rtol=rtol

    kmeans = fit(KMeans(n_clusters=2, n_init=1, max_iterations=100, tolerance=1e-5), X)
    centers = kmeans.centers[sortperm(kmeans.centers[:, 1]), :]
    expected_centers = reduce(vcat, permutedims.(fixture["kmeans"]["centers"]))
    @test centers ≈ expected_centers atol=atol rtol=rtol
    @test kmeans.inertia ≈ fixture["kmeans"]["inertia"] atol=atol rtol=rtol

    for (model, section) in ((GaussianNaiveBayes(), "gaussian_nb"),
                             (LinearDiscriminantAnalysis(), "lda"),
                             (QuadraticDiscriminantAnalysis(), "qda"))
        expected = reduce(vcat, permutedims.(fixture[section]["probabilities"]))
        @test predict_proba(fit(model, X, y), query) ≈ expected atol=atol rtol=rtol
    end

    mixture = fit(GaussianMixture(n_components=2, n_init=1, max_iterations=100,
                                  tolerance=1e-5, regularization=1e-5), X)
    order = sortperm(mixture.means[:, 1])
    expected_means = reduce(vcat, permutedims.(fixture["gaussian_mixture"]["means"]))
    @test mixture.means[order, :] ≈ expected_means atol=atol rtol=rtol
    @test mixture.mixture_weights[order] ≈ fixture["gaussian_mixture"]["weights"] atol=atol rtol=rtol

    neighbors = fit(KNeighborsRegressor(n_neighbors=2, weights=:distance), X, collect(0.0:7.0))
    @test predict(neighbors, query) ≈ fixture["neighbors"]["prediction"] atol=atol rtol=rtol
end
