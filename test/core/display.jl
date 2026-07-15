@testset "REPL and notebook display" begin
    X = Float64[1 2; 3 4; 5 6; 7 8]
    y = [1.0, 2, 3, 4]
    model = RidgeRegression(lambda=0.2)
    fitted = fit(model, X, y)
    graph = fit(Chain(Standardize(), model), X, y)

    compact_model = sprint(show, model)
    plain_model = sprint(show, MIME"text/plain"(), model)
    @test occursin("RidgeRegression", compact_model)
    @test occursin("lambda=0.2", compact_model)
    @test occursin("task: regression", plain_model)

    @test occursin("4×2", sprint(show, fitted))
    plain_fitted = sprint(show, MIME"text/plain"(), fitted)
    @test occursin("4 observations × 2 features", plain_fitted)
    @test !occursin(string(fitted.coefficients), plain_fitted)

    plain_graph = sprint(show, MIME"text/plain"(), graph)
    @test occursin("FittedGraph with 2 nodes", plain_graph)
    @test occursin("FittedStandardize", plain_graph)
    @test occursin("FittedLinearRegressor", plain_graph)

    plain_report = sprint(show, MIME"text/plain"(), report(graph))
    @test occursin("status: success", plain_report)
    @test occursin("fit_execution_graph", plain_report)

    schema_text = sprint(show, MIME"text/plain"(), fitted.schema)
    @test occursin("Schema with 2 features", schema_text)
    @test occursin("x1::Float64", schema_text)

    numerical_graph = report(graph).details.fit_execution_graph
    numerical_text = sprint(show, MIME"text/plain"(), numerical_graph)
    @test occursin("NumericalExecutionGraph [fit]", numerical_text)
    @test occursin("fit_transform", numerical_text)

    execution = Tilia.trace(graph, X)
    trace_text = sprint(show, MIME"text/plain"(), execution)
    @test occursin("ExecutionTrace", trace_text)
    @test occursin("transform", trace_text)
    @test occursin("predict", trace_text)

    validation = cross_validate(model, X, y; cv=KFold(2))
    @test occursin("2 folds", sprint(show, validation))
    importance = permutation_importance(fitted, X, y; n_repeats=2)
    @test sprint(show, importance) ==
          "PermutationImportanceResult(2 features × 2 repeats)"
end
