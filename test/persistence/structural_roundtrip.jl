function _roundtrip_output(fitted, X)
    task = capabilities(fitted.model).task
    task in (:transformation, :neighbors) && return transform(fitted, X)
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
        fit(MeanRegressor(), X, yr),
        fit(Standardize(), X),
        fit(MinMaxScale(), X),
        fit(RobustScale(), X),
        fit(Normalize(), X),
        fit(PolynomialFeatures(degree=2), X),
        fit(LinearRegression(), X, yr),
        fit(RidgeRegression(lambda=0.2), X, yr),
        fit(LogisticRegression(lambda=0.5, max_iterations=30), X, yc),
        fit(PCA(n_components=1), X),
        fit(TruncatedSVD(n_components=1), X),
        fit(KMeans(n_clusters=2, n_init=1), X),
        fit(DBSCAN(radius=2.0, min_neighbors=2), X),
        fit(AgglomerativeClustering(n_clusters=2), X),
        fit(FeatureAgglomeration(n_clusters=1), X),
        fit(GaussianNaiveBayes(), X, yc),
        fit(LinearDiscriminantAnalysis(), X, yc),
        fit(QuadraticDiscriminantAnalysis(), X, yc),
        fit(GaussianMixture(n_components=2, n_init=1), X),
        fit(NearestNeighbors(n_neighbors=2), X),
        fit(KNeighborsClassifier(n_neighbors=2), X, yc),
        fit(KNeighborsRegressor(n_neighbors=2), X, yr),
        fit(Lasso(lambda=0.1), X, yr),
        fit(ElasticNet(lambda=0.1, l1_ratio=0.5), X, yr),
        fit(SparseLogisticRegression(lambda=0.1, max_iterations=20), X, yc),
        fit(SGDClassifier(epochs=2), X, yc),
        fit(SGDRegressor(epochs=2), X, yr),
        fit(MARSRegressor(max_terms=5, max_knots=3), X, yr),
        fit(PartialLeastSquaresRegression(n_components=1), X, yr),
        fit(DecisionTreeClassifier(), X, yc),
        fit(DecisionTreeRegressor(), X, yr),
        fit(RandomForestClassifier(n_estimators=2), X, yc),
        fit(RandomForestRegressor(n_estimators=2), X, yr),
        fit(ExtraTreesClassifier(n_estimators=2), X, yc),
        fit(ExtraTreesRegressor(n_estimators=2), X, yr),
        fit(HistGradientBoostingClassifier(n_estimators=2, min_samples_leaf=1), X, yc),
        fit(HistGradientBoostingRegressor(n_estimators=2, min_samples_leaf=1), X, yr),
        fit(IsolationForest(n_estimators=2), X),
        fit(KernelRidgeRegression(), X, yr),
        fit(SupportVectorClassifier(max_iterations=20), X, yc),
        fit(SupportVectorRegressor(max_iterations=20), X, yr),
        fit(MLPClassifier(hidden_units=3, max_iterations=2), X, yc),
        fit(MLPRegressor(hidden_units=3, max_iterations=2), X, yr),
    )
    for fitted in fitted_models
        _test_structural_roundtrip(fitted, X)
    end

    binary = Float64.(X .> 0)
    nonnegative = abs.(X)
    _test_structural_roundtrip(fit(NMF(n_components=1), nonnegative), nonnegative)
    _test_structural_roundtrip(fit(RandomProjection(n_components=1), X), X)
    _test_structural_roundtrip(fit(FastICA(n_components=1), X), X)
    _test_structural_roundtrip(
        fit(MultinomialNaiveBayes(), nonnegative, yc), nonnegative)
    _test_structural_roundtrip(fit(BernoulliRBM(n_components=2, n_iterations=1), binary), binary)

    selected = fit(Select(1), X)
    _test_structural_roundtrip(selected, X)
    parallel = fit(Parallel(Standardize(), PCA(n_components=1)), X)
    _test_structural_roundtrip(parallel, X)
    branches = transform(parallel, X)
    _test_structural_roundtrip(fit(Concatenate(), branches), branches)

    missing_X = Matrix{Union{Missing,Float64}}(X)
    missing_X[1, 1] = missing
    _test_structural_roundtrip(fit(Impute(), missing_X), missing_X)
    table = column_table((value=X[:, 1], group=yc))
    _test_structural_roundtrip(fit(OneHotEncode(), table), table)
    mapped = fit(ColumnMap(:value => Standardize(),
                           :group => OneHotEncode(passthrough_numeric=false)), table)
    _test_structural_roundtrip(mapped, table)

    mktempdir(pwd()) do directory
        fitted = fit(PCA(n_components=1), X)
        save_model(directory, fitted)
        open(joinpath(directory, "object.toml"), "a") do io
            write(io, "\n# corruption")
        end
        @test_throws Tilia.PersistenceVersionError load_model(directory)
    end
end

@testset "Persistence format migrations" begin
    version_one = Dict{String,Any}(
        "format_version" => 1,
        "estimator" => "GenericFittedEstimator",
        "checksums" => Dict{String,String}(),
    )
    version_two = Tilia.migrate(Val(1), Val(2), version_one)
    @test version_two["format_version"] == 2
    @test version_two["estimator_schema_version"] == 1
    @test version_two["migration_history"] == ["1=>2"]
    @test version_one["format_version"] == 1
    @test Tilia._parse_persisted_float(Float32, "-0.29166722f0") == -0.29166722f0

    X = Float32[1 2; 3 4; 5 6]
    y = Float32[1, 2, 4]
    fitted = fit(LinearRegression(), X, y)
    mktempdir(pwd()) do directory
        save_model(directory, fitted)
        manifest_path = joinpath(directory, "manifest.toml")
        manifest = TOML.parsefile(manifest_path)
        manifest["format_version"] = 1
        open(manifest_path, "w") do io
            TOML.print(io, manifest; sorted=true)
        end
        loaded = load_model(directory)
        @test predict(loaded, X) ≈ predict(fitted, X)
    end

    @test_throws Tilia.PersistenceVersionError Tilia._migrate_manifest(
        Dict("format_version" => 0))
    @test_throws Tilia.PersistenceVersionError Tilia._migrate_manifest(
        Dict("format_version" => 3))
end
