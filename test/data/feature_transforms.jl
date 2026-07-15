@testset "Common numeric feature transforms" begin
    X = Float64[1 2; 3 4; 5 2]

    scaled = fit(MinMaxScale(feature_range=(-1.0, 1.0)), X)
    transformed = transform(scaled, X)
    @test transformed ≈ [-1 -1; 0 1; 1 -1]
    @test inverse_transform(scaled, transformed) ≈ X
    @test report(scaled).details.constant_features == 0
    clipped = fit(MinMaxScale(clip=true), X)
    @test transform(clipped, [10.0 -2.0]) == [1.0 0.0]

    robust_X = Float64[1 10; 2 20; 3 30; 100 40]
    robust = fit(RobustScale(), robust_X)
    robust_result = transform(robust, robust_X)
    @test robust.medians ≈ [2.5, 25.0]
    @test robust.scales ≈ [25.5, 15.0]
    @test inverse_transform(robust, robust_result) ≈ robust_X
    @test transform(fit(RobustScale(center=false), robust_X), robust_X)[:, 1] ≈
          robust_X[:, 1] ./ 25.5
    constant_robust = fit(RobustScale(), ones(4, 1))
    @test transform(constant_robust, ones(4, 1)) == zeros(4, 1)

    normalized = fit(Normalize(norm=:l2), X)
    normalized_X = transform(normalized, X)
    @test vec(sqrt.(sum(abs2, normalized_X; dims=2))) ≈ ones(3)
    sparse_X = sparse([3.0 0; 0 4; 0 0])
    sparse_result = transform(fit(Normalize(), sparse_X), sparse_X)
    @test sparse_result isa SparseMatrixCSC
    @test Matrix(sparse_result) == [1.0 0; 0 1; 0 0]

    polynomial = fit(PolynomialFeatures(degree=2), X[:, 1:2])
    @test transform(polynomial, X[:, 1:2]) == [
        1 1 2 1 2 4
        1 3 4 9 12 16
        1 5 2 25 10 4
    ]
    interactions = fit(PolynomialFeatures(degree=3, include_bias=false,
                                           interaction_only=true), X[:, 1:2])
    @test transform(interactions, X[:, 1:2]) == [1 2 2; 3 4 12; 5 2 10]
    @test report(polynomial).details.output_features == 6

    y = 2 .* X[:, 1] .+ X[:, 2]
    pipeline = fit(Chain(PolynomialFeatures(degree=2, include_bias=false),
                         RidgeRegression(lambda=0.1)), X, y)
    @test length(predict(pipeline, X)) == size(X, 1)

    @test eltype(transform(fit(MinMaxScale(), Float32.(X)), Float32.(X))) == Float32
    @test eltype(transform(fit(RobustScale(), Float32.(X)), Float32.(X))) == Float32
    @test eltype(transform(fit(PolynomialFeatures(), Float32.(X)), Float32.(X))) == Float32
    @test_throws Tilia.InvalidHyperparameterError MinMaxScale(feature_range=(1, 1))
    @test_throws Tilia.InvalidHyperparameterError RobustScale(quantile_range=(0.8, 0.2))
    @test_throws Tilia.InvalidHyperparameterError Normalize(norm=:bad)
    @test_throws Tilia.InvalidHyperparameterError PolynomialFeatures(degree=0)
    @test_throws Tilia.InvalidHyperparameterError PolynomialFeatures(degree=33)
    @test_throws Tilia.UnsupportedDataError fit(
        PolynomialFeatures(degree=5), ones(1, 50))
end
