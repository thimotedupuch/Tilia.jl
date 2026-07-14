@testset "Logistic regression" begin
    X = [-2.0 0.0; -1.0 1.0; 0.0 -1.0; 0.5 0.5; 1.0 -0.5; 2.0 1.0]
    y = ["no", "no", "no", "yes", "yes", "yes"]
    fitted = fit(LogisticRegression(lambda=0.5), X, y)
    probabilities = predict_proba(fitted, X)

    @test fitted.classes == ["no", "yes"]
    @test size(probabilities) == (6, 2)
    @test vec(sum(probabilities; dims=2)) ≈ ones(6) atol=1e-12
    @test all(probability -> 0 <= probability <= 1, probabilities)
    @test predict(fitted, X) == y
    @test report(fitted).status == :success
    @test report(fitted).details.class_order == ["no", "yes"]
    @test issorted(reverse(only(report(fitted).details.objective_history)))
    @test report(fitted).details.gradient_norms[1] <=
          fitted.model.tolerance * (1 + maximum(abs, [fitted.coefficients[:, 1]; fitted.intercept[1]]))

    # Weighting agrees with explicitly duplicating observations for integer weights.
    weights = [1, 2, 1, 2, 1, 1]
    weighted = fit(LogisticRegression(lambda=0.5), X, y; weights=weights)
    duplicated_indices = reduce(vcat, fill(index, weight) for (index, weight) in enumerate(weights))
    duplicated = fit(LogisticRegression(lambda=0.5), X[duplicated_indices, :], y[duplicated_indices])
    @test weighted.coefficients ≈ duplicated.coefficients atol=1e-10
    @test weighted.intercept ≈ duplicated.intercept atol=1e-10

    multiclass_X = [-2.0 0.0; -1.5 0.2; 2.0 0.0; 1.5 -0.2; 0.0 2.0; 0.2 1.5]
    multiclass_y = [:left, :left, :right, :right, :up, :up]
    multiclass = fit(LogisticRegression(lambda=0.2), multiclass_X, multiclass_y)
    multiclass_probabilities = predict_proba(multiclass, multiclass_X)
    @test multiclass.classes == [:left, :right, :up]
    @test size(multiclass_probabilities) == (6, 3)
    @test vec(sum(multiclass_probabilities; dims=2)) ≈ ones(6) atol=1e-12
    @test predict(multiclass, multiclass_X) == multiclass_y

    for T in (Float32, Float64)
        typed = fit(LogisticRegression(lambda=1.0), T.(X), y)
        @test eltype(typed.coefficients) == T
        @test eltype(predict_proba(typed, T.(X))) == T
    end

    unfinished = fit(LogisticRegression(max_iterations=1, tolerance=1e-15), X, y)
    @test report(unfinished).status == :max_iterations
    @test !isempty(report(unfinished).warnings)
    @test_throws Tilia.InvalidHyperparameterError LogisticRegression(lambda=-1)
    @test_throws Tilia.InvalidHyperparameterError LogisticRegression(max_iterations=0)
    @test_throws Tilia.UnsupportedDataError fit(LogisticRegression(), X, fill("yes", 6))
    @test_throws Tilia.SchemaMismatchError predict_proba(fitted, ones(2, 3))
end
