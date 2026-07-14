function _roundtrip_output(fitted, X)
    task = capabilities(fitted.model).task
    task === :transformation && return transform(fitted, X)
    task === :anomaly_detection && return anomaly_score(fitted, X)
    capabilities(fitted.model).probabilistic && return predict_proba(fitted, X)
    predict(fitted, X)
end

function _test_structural_roundtrip(fitted, X)
    expected = _roundtrip_output(fitted, X)
    mktempdir(pwd()) do directory
        save_model(directory, fitted)
        loaded = load_model(directory)
        @test typeof(loaded.model) == typeof(fitted.model)
        actual = _roundtrip_output(loaded, X)
        @test eltype(expected) <: Number ? actual ≈ expected : actual == expected
        @test report(loaded).status == report(fitted).status
    end
end

@testset "Structural persistence round trips" begin
    X = [-2.0 -1; -1 -2; 1 2; 2 1; 2.5 2; -2.5 -2]
    yr = [1.0, 2, -1, -2, -2.5, 2.5]
    yc = [:left, :left, :right, :right, :right, :left]

    fitted_models = (
        fit(PCA(n_components=1), X),
        fit(KMeans(n_clusters=2, n_init=1), X),
        fit(GaussianNaiveBayes(), X, yc),
        fit(LinearDiscriminantAnalysis(), X, yc),
        fit(QuadraticDiscriminantAnalysis(), X, yc),
        fit(GaussianMixture(n_components=2, n_init=1), X),
        fit(KNeighborsClassifier(n_neighbors=2), X, yc),
        fit(Lasso(lambda=0.1), X, yr),
        fit(SparseLogisticRegression(lambda=0.1, max_iterations=20), X, yc),
        fit(DecisionTreeClassifier(), X, yc),
        fit(RandomForestRegressor(n_estimators=2), X, yr),
        fit(HistGradientBoostingRegressor(n_estimators=2, min_samples_leaf=1), X, yr),
        fit(IsolationForest(n_estimators=2), X),
        fit(KernelRidgeRegression(), X, yr),
        fit(SupportVectorClassifier(max_iterations=20), X, yc),
        fit(SupportVectorRegressor(max_iterations=20), X, yr),
        fit(MLPClassifier(hidden_units=3, max_iterations=2), X, yc),
    )
    for fitted in fitted_models
        _test_structural_roundtrip(fitted, X)
    end

    binary = Float64.(X .> 0)
    _test_structural_roundtrip(fit(BernoulliRBM(n_components=2, n_iterations=1), binary), binary)

    mktempdir(pwd()) do directory
        fitted = fit(PCA(n_components=1), X)
        save_model(directory, fitted)
        open(joinpath(directory, "object.toml"), "a") do io
            write(io, "\n# corruption")
        end
        @test_throws Tilia.PersistenceVersionError load_model(directory)
    end
end
