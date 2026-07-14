@testset "Kernel functions and ridge regression" begin
    X = [-2.0 -1; -1 -2; 1 2; 2 1]
    linear = Tilia.Kernels.gram_matrix(X; kernel=:linear)
    @test linear == X * transpose(X)
    rbf = Tilia.Kernels.gram_matrix(X; kernel=:rbf, gamma=0.5)
    @test rbf ≈ transpose(rbf)
    @test diag(rbf) == ones(4)
    @test all(eigvals(Symmetric(rbf)) .>= -1e-12)
    polynomial = Tilia.Kernels.gram_matrix(X; kernel=:polynomial,
                                           gamma=0.5, degree=2, coef0=1)
    @test polynomial ≈ (0.5 .* linear .+ 1) .^ 2

    y = [1.0, 2.0, -1.0, -2.0]
    for kernel in (:linear, :rbf, :polynomial)
        model = KernelRidgeRegression(lambda=0.01, kernel=kernel, gamma=0.5, degree=2)
        fitted = fit(model, X, y)
        @test root_mean_squared_error(y, predict(fitted, X)) < 0.6
        @test report(fitted).details.kernel == kernel
    end
    rbf_fit = fit(KernelRidgeRegression(lambda=0.01, kernel=:rbf, gamma=0.5), X, y)
    @test root_mean_squared_error(y, predict(rbf_fit, X)) < 0.1

    X32, y32 = Float32.(X), Float32.(y)
    fitted32 = fit(KernelRidgeRegression(lambda=0.1), X32, y32)
    @test eltype(predict(fitted32, X32)) == Float32
    @test_throws Tilia.InvalidHyperparameterError KernelRidgeRegression(lambda=0)
    @test_throws Tilia.InvalidHyperparameterError KernelRidgeRegression(kernel=:bad)
    @test_throws Tilia.UnsupportedDataError fit(KernelRidgeRegression(), X, y; weights=ones(4))
end
