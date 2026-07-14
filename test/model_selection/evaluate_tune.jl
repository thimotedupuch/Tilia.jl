@testset "Evaluation and tuning" begin
    X = reshape(collect(1.0:12.0), :, 1)
    y = @. 2 * X[:, 1] + 1
    cv = KFold(3; shuffle=true, seed=42)
    evaluation = evaluate(RidgeRegression(), X, y; cv=cv)
    direct = cross_validate(RidgeRegression(), X, y; cv=cv)
    @test evaluation.scores == direct.scores
    @test evaluation.train_indices == direct.train_indices

    result = tune(RidgeRegression(), X, y;
        parameter_grid=(lambda=[0.0, 0.1, 10.0], solver=[:cholesky],), cv=cv)
    @test result isa TuningResult
    @test length(result.trials) == 3
    @test result.best_parameters.lambda in (0.0, 0.1, 10.0)
    @test result.best_score == minimum(trial.score for trial in result.trials)
    @test result.fitted_model !== nothing
    @test size(predict(result.fitted_model, X)) == (12,)

    y_class = ifelse.(X[:, 1] .> 6, :high, :low)
    classification = tune(LogisticRegression(max_iterations=200), X, y_class;
        parameter_grid=(lambda=[0.01, 1.0],), cv=KFold(3))
    @test classification.best_score == maximum(trial.score for trial in classification.trials)
    @test classification.best_model.lambda in (0.01, 1.0)

    no_refit = tune(RidgeRegression(), X, y;
        parameter_grid=(lambda=[0.1],), cv=cv, refit=false)
    @test no_refit.fitted_model === nothing
    @test_throws Tilia.InvalidHyperparameterError tune(RidgeRegression(), X, y;
        parameter_grid=(unknown=[1],), cv=cv)
    @test_throws Tilia.InvalidHyperparameterError tune(RidgeRegression(), X, y;
        parameter_grid=NamedTuple(), cv=cv)
end
