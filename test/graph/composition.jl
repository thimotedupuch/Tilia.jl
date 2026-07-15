@testset "Parallel, column mapping, selection, and concatenation" begin
    X = [1.0 10 0; 2 20 1; 3 30 0; 4 40 1]
    selected = fit(Select(1, 3), X)
    @test transform(selected, X) == X[:, [1, 3]]
    @test report(selected).details.selected_columns == [1, 3]

    parallel = fit(Parallel(Standardize(), PCA(n_components=1)), X)
    branches = transform(parallel, X)
    @test length(branches) == 2
    @test size(branches[1]) == (4, 3)
    @test size(branches[2]) == (4, 1)
    concatenated = fit(Concatenate(), branches)
    @test transform(concatenated, branches) == hcat(branches...)

    y = [1.0, 2.0, 3.0, 4.0]
    graph_model = Chain(Parallel(Standardize(), PCA(n_components=1)),
                        Concatenate(), RidgeRegression(lambda=0.1))
    graph_fit = fit(graph_model, X, y)
    @test size(predict(graph_fit, X)) == (4,)
    @test report(graph_fit).details.nodes == 4
    @test graph_fit.graph.edges == [(1, 3), (2, 3), (3, 4)]
    @test graph_fit.graph.nodes[1].model isa Standardize
    @test graph_fit.graph.nodes[2].model isa PCA
    @test graph_fit.graph.nodes[3].model isa Concatenate
    @test graph_fit.graph.nodes[4].model isa RidgeRegression
    @test Tilia.optimize(graph_fit) === graph_fit
    mktempdir(pwd()) do directory
        save_model(directory, graph_fit)
        loaded = load_model(directory)
        @test predict(loaded, X) ≈ predict(graph_fit, X)
        @test length(loaded.fitted_nodes) == 4
    end

    table = column_table((age=[20.0, 22.0, 40.0, 42.0],
                          color=[:blue, :blue, :red, :red]))
    table_select = fit(Select(:color), table)
    selected_table = transform(table_select, table)
    @test selected_table.names == (:color,)
    @test size(selected_table) == (4, 1)

    mapped = ColumnMap(:age => Standardize(),
                       :color => OneHotEncode(passthrough_numeric=false))
    mapped_fit = fit(mapped, table)
    mapped_values = transform(mapped_fit, table)
    @test size(mapped_values) == (4, 3)
    @test vec(mean(mapped_values[:, 1]; dims=1)) ≈ [0.0] atol=1e-12

    classifier = fit(Chain(mapped, LogisticRegression(lambda=0.1)),
                     table, [:young, :young, :old, :old])
    @test length(classifier.graph.nodes) == 6
    @test classifier.graph.edges == [(1, 2), (3, 4), (2, 5), (4, 5), (5, 6)]
    @test predict(classifier, table) == [:young, :young, :old, :old]
    @test size(predict_proba(classifier, table)) == (4, 2)
    mktempdir(pwd()) do directory
        save_model(directory, classifier)
        loaded = load_model(directory)
        @test predict_proba(loaded, table) ≈ predict_proba(classifier, table)
    end

    @test_throws Tilia.InvalidHyperparameterError Parallel()
    @test_throws Tilia.InvalidHyperparameterError ColumnMap(:age => MeanRegressor())
    @test_throws Tilia.SchemaMismatchError transform(selected, ones(2, 4))
end


@testset "Static semantic schema propagation" begin
    matrix_schema = Tilia.with_target(Tilia.infer_schema(randn(8, 3)), randn(8))
    decomposition = Tilia.build_graph(Chain(PCA(n_components=2), RidgeRegression()))
    schemas = Tilia.propagate_schema(decomposition, matrix_schema; observations=8)
    @test [column.name for column in schemas[1].columns] == [:component1, :component2]
    @test only(schemas[2].columns).role == :prediction

    branched = Tilia.build_graph(Chain(
        Parallel(Standardize(), PCA(n_components=1)), Concatenate(), MeanRegressor()))
    branch_schemas = Tilia.propagate_schema(branched, matrix_schema; observations=8)
    @test Tilia.nfeatures(branch_schemas[1]) == 3
    @test Tilia.nfeatures(branch_schemas[2]) == 1
    @test Tilia.nfeatures(branch_schemas[3]) == 4

    table = column_table((value=[1.0, 2, 3], group=[:a, :b, :a]))
    mapped = ColumnMap(:value => Standardize(),
                       :group => OneHotEncode(passthrough_numeric=false))
    mapped_schema = output_schema(mapped, table.schema)
    @test [column.name for column in mapped_schema.columns] ==
          [:value, :group__a, :group__b]
    @test all(column -> !isempty(column.provenance), mapped_schema.columns)

    missing_schema = Tilia.infer_schema(Matrix{Union{Missing,Float64}}(
        [1.0 2.0; 3.0 4.0]))
    @test all(!column.allows_missing for column in
              output_schema(Impute(), missing_schema).columns)
    @test !capabilities(Parallel(Standardize(), Impute())).missing
    @test capabilities(Parallel(Impute(), Impute())).missing
    @test !capabilities(Concatenate()).missing
    @test capabilities(Select(1)).sparse
    @test !capabilities(Chain(Standardize(), Lasso())).sparse
    @test capabilities(Chain(Impute(), RidgeRegression())).missing
    @test output_schema(RobustScale(), matrix_schema) == matrix_schema

    polynomial_schema = output_schema(
        PolynomialFeatures(degree=2), matrix_schema)
    @test Tilia.nfeatures(polynomial_schema) == 10
    @test polynomial_schema.columns[1].name == :bias
    @test polynomial_schema.columns[2].provenance == [:x1]
end
