@testset "Initial-scope scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "initial_scope_sklearn.toml"))
    @test fixture["source"]["package"] == "scikit-learn"
    @test fixture["source"]["version"] == "1.9.0"
    atol = fixture["tolerance"]["absolute"]
    rtol = fixture["tolerance"]["relative"]
    X = reduce(vcat, permutedims.(fixture["input"]["X"]))
    query = reduce(vcat, permutedims.(fixture["input"]["query"]))
    y_reg = fixture["input"]["y_reg"]
    y_class = fixture["input"]["y_class"]

    for (model, section) in ((LinearRegression(), "linear_regression"),
                             (RidgeRegression(lambda=1.0), "ridge"),
                             (Lasso(lambda=0.05, max_iterations=10_000,
                                    tolerance=1e-10), "lasso"))
        fitted = fit(model, X, y_reg)
        @test fitted.coefficients ≈ fixture[section]["coefficients"] atol=atol rtol=rtol
        @test fitted.intercept ≈ fixture[section]["intercept"] atol=atol rtol=rtol
        @test predict(fitted, query) ≈ fixture[section]["prediction"] atol=atol rtol=rtol
    end

    logistic = fit(LogisticRegression(lambda=1.0), X, y_class)
    @test predict(logistic, query) == fixture["logistic"]["prediction"]

    neighbors = fit(NearestNeighbors(n_neighbors=3), X)
    distances, indices = kneighbors(neighbors, query)
    expected_distances = reduce(vcat, permutedims.(fixture["neighbors"]["distances"]))
    expected_indices = reduce(vcat, permutedims.(fixture["neighbors"]["indices_zero_based"])) .+ 1
    @test distances ≈ expected_distances atol=atol rtol=rtol
    @test indices == expected_indices

    tree = fit(DecisionTreeClassifier(max_depth=3), X, y_class)
    @test predict(tree, query) == fixture["decision_tree"]["prediction"]

    kernel = fit(KernelRidgeRegression(lambda=0.2, kernel=:rbf, gamma=0.5), X, y_reg)
    @test predict(kernel, query) ≈ fixture["kernel_ridge"]["prediction"] atol=atol rtol=rtol

    support = fit(SupportVectorClassifier(C=2.0, kernel=:linear,
                                          max_iterations=2_000), X, y_class)
    @test predict(support, query) == fixture["support_vector"]["prediction"]
end
