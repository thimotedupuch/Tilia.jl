const NUMERICAL_CONFORMANCE_CASES = (
    (type=MeanRegressor, make=T -> MeanRegressor()),
    (type=Standardize, make=T -> Standardize()),
    (type=LinearRegression, make=T -> LinearRegression()),
    (type=RidgeRegression, make=T -> RidgeRegression(lambda=T(0.2))),
    (type=LogisticRegression, make=T -> LogisticRegression(lambda=T(0.5), max_iterations=30)),
    (type=Lasso, make=T -> Lasso(lambda=T(0.05), max_iterations=50)),
    (type=ElasticNet, make=T -> ElasticNet(lambda=T(0.05), l1_ratio=T(0.5), max_iterations=50)),
    (type=SparseLogisticRegression, make=T -> SparseLogisticRegression(lambda=T(0.05), max_iterations=50)),
    (type=PCA, make=T -> PCA(n_components=2)),
    (type=TruncatedSVD, make=T -> TruncatedSVD(n_components=2)),
    (type=KMeans, make=T -> KMeans(n_clusters=2, n_init=1, max_iterations=20)),
    (type=GaussianNaiveBayes, make=T -> GaussianNaiveBayes()),
    (type=LinearDiscriminantAnalysis, make=T -> LinearDiscriminantAnalysis()),
    (type=QuadraticDiscriminantAnalysis, make=T -> QuadraticDiscriminantAnalysis()),
    (type=NearestNeighbors, make=T -> NearestNeighbors(n_neighbors=3)),
    (type=KNeighborsClassifier, make=T -> KNeighborsClassifier(n_neighbors=3)),
    (type=KNeighborsRegressor, make=T -> KNeighborsRegressor(n_neighbors=3)),
    (type=GaussianMixture, make=T -> GaussianMixture(n_components=2, n_init=1, max_iterations=20)),
    (type=DecisionTreeClassifier, make=T -> DecisionTreeClassifier(max_depth=3)),
    (type=DecisionTreeRegressor, make=T -> DecisionTreeRegressor(max_depth=3)),
    (type=RandomForestClassifier, make=T -> RandomForestClassifier(n_estimators=2, max_depth=3)),
    (type=RandomForestRegressor, make=T -> RandomForestRegressor(n_estimators=2, max_depth=3)),
    (type=ExtraTreesClassifier, make=T -> ExtraTreesClassifier(n_estimators=2, max_depth=3)),
    (type=ExtraTreesRegressor, make=T -> ExtraTreesRegressor(n_estimators=2, max_depth=3)),
    (type=HistGradientBoostingClassifier, make=T -> HistGradientBoostingClassifier(
        n_estimators=2, max_depth=2, min_samples_leaf=2)),
    (type=HistGradientBoostingRegressor, make=T -> HistGradientBoostingRegressor(
        n_estimators=2, max_depth=2, min_samples_leaf=2)),
    (type=IsolationForest, make=T -> IsolationForest(n_estimators=2, max_samples=8)),
    (type=KernelRidgeRegression, make=T -> KernelRidgeRegression(lambda=T(0.2))),
    (type=SupportVectorClassifier, make=T -> SupportVectorClassifier(
        C=T(2), kernel=:linear, max_iterations=20, tolerance=T(1e-5))),
    (type=SupportVectorRegressor, make=T -> SupportVectorRegressor(
        C=T(2), kernel=:linear, max_iterations=20, tolerance=T(1e-5))),
    (type=MLPClassifier, make=T -> MLPClassifier(hidden_units=4, max_iterations=2)),
    (type=MLPRegressor, make=T -> MLPRegressor(hidden_units=4, max_iterations=2)),
    (type=BernoulliRBM, make=T -> BernoulliRBM(n_components=2, batch_size=4, n_iterations=1)),
)

function _conformance_data(::Type{T}, model) where {T}
    X = T[sin(i * (j + 0.3)) + cos((i + 2j) / 3) for i in 1:16, j in 1:4]
    model isa BernoulliRBM && (X = T.(X .> 0))
    yreg = T[0.7X[i, 1] - 0.2X[i, 2] + 0.1i for i in axes(X, 1)]
    yclass = [i <= 8 ? :negative : :positive for i in axes(X, 1)]
    X, yreg, yclass
end

function _conformance_fit(model, X, yreg, yclass; weights=nothing)
    task = capabilities(model).task
    if task === :regression
        return fit(model, X, yreg; weights=weights)
    elseif task === :classification
        return fit(model, X, yclass; weights=weights)
    end
    weights === nothing ? fit(model, X) : fit(model, X; weights)
end

function _conformance_output(fitted, X)
    task = capabilities(fitted.model).task
    task in (:transformation, :neighbors) ? transform(fitted, X) : predict(fitted, X)
end

_outputs_match(left, right) = left isa AbstractArray && eltype(left) <: Number ?
    isapprox(left, right; rtol=1e-4, atol=1e-5) : left == right

@testset "Universal numerical estimator conformance" begin
    special_types = Set((Impute, OneHotEncode, Select, Parallel, ColumnMap, Concatenate))
    registered = Set(entry.type for entry in model_catalog())
    covered = Set(case.type for case in NUMERICAL_CONFORMANCE_CASES)
    @test union(covered, special_types) == registered
    @test isempty(intersect(covered, special_types))

    for case in NUMERICAL_CONFORMANCE_CASES
        @testset "$(case.type)" begin
            for T in (Float32, Float64)
                model = case.make(T)
                X, yreg, yclass = _conformance_data(T, model)
                original_X = copy(X)
                original_yreg = copy(yreg)
                original_yclass = copy(yclass)
                first_fit = _conformance_fit(model, X, yreg, yclass)
                second_fit = _conformance_fit(case.make(T), X, yreg, yclass)
                first_output = _conformance_output(first_fit, X)
                second_output = _conformance_output(second_fit, X)

                @test size(first_output, 1) == size(X, 1)
                @test _outputs_match(first_output, second_output)
                declared = capabilities(model)
                declared.task in (:regression, :transformation, :neighbors) &&
                    eltype(first_output) <: Number && @test eltype(first_output) == T
                @test X == original_X
                @test yreg == original_yreg
                @test yclass == original_yclass
                @test report(first_fit) isa Tilia.FitReport
                @test report(first_fit).observations == size(X, 1)
                @test report(first_fit).features == size(X, 2)
                @test report(first_fit).backend == :cpu
                @test report(first_fit).root_seed == UInt64(0)
                @test report(first_fit).stream_id == "root"
                @test report(first_fit).deterministic
                @test report(first_fit).thread_count == Threads.nthreads()
                @test hasproperty(report(first_fit).details, :numerical_policy)
                @test report(first_fit).details.numerical_policy.float_type == "Float64"
                if declared.task in (:classification, :regression)
                    @test first_fit.schema.target_name == :target
                    @test first_fit.schema.target_logical_type ==
                          (declared.task === :classification ? :categorical : :continuous)
                    @test first_fit.schema.target_physical_type ==
                          (declared.task === :classification ? Symbol : T)
                    @test !first_fit.schema.target_allows_missing
                end

                if declared.task === :classification && declared.probabilistic
                    probabilities = predict_proba(first_fit, X)
                    @test size(probabilities, 1) == size(X, 1)
                    @test eltype(probabilities) == T
                    @test vec(sum(probabilities; dims=2)) ≈ ones(T, size(X, 1)) rtol=1e-4
                end
                if declared.task === :classification
                    @test hasproperty(first_fit, :classes)
                    @test first_fit.classes == sort(unique(yclass))
                end
                if declared.sparse
                    sparse_fit = _conformance_fit(case.make(T), sparse(X), yreg, yclass)
                    sparse_output = _conformance_output(sparse_fit, sparse(X))
                    @test _outputs_match(first_output, sparse_output)
                end
                if declared.task in (:classification, :regression)
                    bad_target = declared.task === :classification ? yclass[1:end-1] : yreg[1:end-1]
                    @test_throws Union{Tilia.SchemaMismatchError,Tilia.UnsupportedDataError} fit(
                        case.make(T), X, bad_target)
                end
                if declared.weights && declared.task in (:classification, :regression)
                    weighted = _conformance_fit(case.make(T), X, yreg, yclass;
                                                weights=ones(T, size(X, 1)))
                    @test _outputs_match(first_output, _conformance_output(weighted, X))
                end
                if !declared.weights
                    @test_throws Tilia.UnsupportedDataError _conformance_fit(
                        case.make(T), X, yreg, yclass;
                        weights=ones(T, size(X, 1)))
                end
                if T === Float64
                    empty_X = Matrix{T}(undef, 0, size(X, 2))
                    @test_throws Tilia.TiliaError _conformance_fit(
                        case.make(T), empty_X, T[], Symbol[])
                end
                if declared.partial_fit
                    if model isa MeanRegressor
                        online = partial_fit(case.make(T), X[1:8, :], yreg[1:8])
                        online = partial_fit(online, X[9:16, :], yreg[9:16])
                        @test predict(online, X) ≈ predict(first_fit, X) rtol=1e-4
                    elseif model isa Standardize
                        online = partial_fit(case.make(T), X[1:8, :])
                        online = partial_fit(online, X[9:16, :])
                        @test transform(online, X) ≈ transform(first_fit, X) rtol=1e-4
                    end
                end

                graph_model = declared.task in (:transformation, :neighbors) ?
                    Chain(case.make(T), Standardize()) :
                    Chain(Standardize(), case.make(T))
                graph_fit = if declared.task === :regression
                    fit(graph_model, X, yreg)
                elseif declared.task === :classification
                    fit(graph_model, X, yclass)
                else
                    fit(graph_model, X)
                end
                repeated_graph_fit = if declared.task === :regression
                    fit(Chain(Standardize(), case.make(T)), X, yreg)
                elseif declared.task === :classification
                    fit(Chain(Standardize(), case.make(T)), X, yclass)
                else
                    repeated_model = declared.task in (:transformation, :neighbors) ?
                        Chain(case.make(T), Standardize()) :
                        Chain(Standardize(), case.make(T))
                    fit(repeated_model, X)
                end
                graph_output = declared.task in (:transformation, :neighbors) ?
                    transform(graph_fit, X) : predict(graph_fit, X)
                repeated_graph_output = declared.task in (:transformation, :neighbors) ?
                    transform(repeated_graph_fit, X) : predict(repeated_graph_fit, X)
                @test size(graph_output, 1) == size(X, 1)
                @test _outputs_match(graph_output, repeated_graph_output)
            end
        end
    end
end

@testset "Heterogeneous transformer conformance" begin
    for T in (Float32, Float64)
        missing_X = Matrix{Union{Missing,T}}(T[1 2; 3 4; 5 6])
        missing_X[2, 1] = missing
        missing_copy = copy(missing_X)
        first_imputer = fit(Impute(), missing_X)
        second_imputer = fit(Impute(), missing_X)
        @test transform(first_imputer, missing_X) == transform(second_imputer, missing_X)
        @test isequal(missing_X, missing_copy)
        @test eltype(transform(first_imputer, missing_X)) == T
        @test report(first_imputer).observations == 3

        table = column_table((value=T[1, 2, 3, 4], group=[:a, :b, :a, :b]))
        encoder = fit(OneHotEncode(output_type=T), table)
        encoded = transform(encoder, table)
        @test size(encoded) == (4, 3)
        @test eltype(encoded) == T
        @test encoded == transform(fit(OneHotEncode(output_type=T), table), table)

        X = T[1 10 0; 2 20 1; 3 30 0; 4 40 1]
        original = copy(X)
        selected = fit(Select(1, 3), X)
        @test transform(selected, X) == X[:, [1, 3]]
        @test X == original

        parallel = fit(Parallel(Standardize(), PCA(n_components=1)), X)
        branches = transform(parallel, X)
        @test map(size, branches) == ((4, 3), (4, 1))
        concatenated = fit(Concatenate(), branches)
        @test transform(concatenated, branches) == hcat(branches...)

        mapped = ColumnMap(:value => Standardize(),
                           :group => OneHotEncode(output_type=T,
                                                 passthrough_numeric=false))
        first_mapped = fit(mapped, table)
        second_mapped = fit(mapped, table)
        @test transform(first_mapped, table) ≈ transform(second_mapped, table)
        @test size(transform(first_mapped, table)) == (4, 3)

        weighted_inputs = (
            (Impute(), missing_X),
            (OneHotEncode(output_type=T), table),
            (Select(1), X),
            (Parallel(Standardize(), PCA(n_components=1)), X),
            (Concatenate(), branches),
            (mapped, table),
        )
        for (model, input) in weighted_inputs
            @test !capabilities(model).weights
            @test_throws Tilia.UnsupportedDataError fit(
                model, input; weights=ones(T, size(input, 1)))
        end
        @test_throws Tilia.UnsupportedDataError partial_fit(
            first_imputer, missing_X; weights=ones(T, size(missing_X, 1)))
        @test_throws Tilia.UnsupportedDataError partial_fit(
            fit(Standardize(), X), X; weights=ones(T, size(X, 1)))

        for fitted in (first_imputer, encoder, selected, parallel, concatenated, first_mapped)
            @test report(fitted) isa Tilia.FitReport
            @test report(fitted).backend == :cpu
        end
    end
end
