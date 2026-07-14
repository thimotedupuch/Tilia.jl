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
    @test report(graph_fit).details.nodes == 3
    mktempdir(pwd()) do directory
        save_model(directory, graph_fit)
        loaded = load_model(directory)
        @test predict(loaded, X) ≈ predict(graph_fit, X)
        @test length(loaded.fitted_nodes) == 3
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
