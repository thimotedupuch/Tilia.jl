function basic_regressor_conformance(model)
    for T in (Float32, Float64)
        X = T[1 2; 3 4; 5 6]
        y = T[1, 4, 7]
        Xcopy, ycopy = copy(X), copy(y)
        first_fit = fit(model, X, y)
        second_fit = fit(model, X, y)
        @test size(predict(first_fit, X)) == (size(X, 1),)
        @test predict(first_fit, X) == predict(second_fit, X)
        @test X == Xcopy
        @test y == ycopy
        @test eltype(predict(first_fit, X)) == T
        @test report(first_fit).status == :success
    end
end

@testset "Estimator conformance" begin
    basic_regressor_conformance(MeanRegressor())
    basic_regressor_conformance(Chain(Standardize(), MeanRegressor()))
    basic_regressor_conformance(LinearRegression())
    basic_regressor_conformance(RidgeRegression())
    @test capabilities(MeanRegressor()).task == :regression
    @test capabilities(Standardize()).task == :transformation

    X = [-2.0 -1.0; -1.0 -2.0; 1.0 2.0; 2.0 1.0]
    y = [:negative, :negative, :positive, :positive]
    pipeline = fit(Chain(PCA(n_components=1), KNeighborsClassifier(n_neighbors=1)), X, y)
    @test predict(pipeline, X) == y
    @test size(predict_proba(pipeline, X)) == (4, 2)
    @test_throws Tilia.UnsupportedDataError predict_proba(
        fit(MeanRegressor(), X, Float64[1, 2, 3, 4]), X)
    @test_throws Tilia.UnsupportedDataError partial_fit(
        RidgeRegression(), X, Float64[1, 2, 3, 4])
    @test_throws Tilia.UnsupportedDataError fit(RidgeRegression(), X)
end
