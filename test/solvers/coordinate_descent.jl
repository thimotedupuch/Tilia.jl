@testset "Coordinate descent solver" begin
    X = [1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0]
    y = X * [2.0, -1.0] .+ 0.5
    result = Tilia.Solvers.elastic_net_coordinate_descent(X, y;
        l1_penalty=0.0, l2_penalty=0.0, tolerance=1e-10, max_iterations=10_000)
    @test result.converged
    @test result.coefficients ≈ [2.0, -1.0] atol=1e-7
    @test result.intercept ≈ 0.5 atol=1e-7
    @test all(diff(result.objective_history) .<= 100eps())

    sparse_result = Tilia.Solvers.elastic_net_coordinate_descent(sparse(X), y;
        l1_penalty=0.1, l2_penalty=0.2, tolerance=1e-10)
    dense_result = Tilia.Solvers.elastic_net_coordinate_descent(X, y;
        l1_penalty=0.1, l2_penalty=0.2, tolerance=1e-10)
    @test sparse_result.coefficients ≈ dense_result.coefficients
    @test sparse_result.intercept ≈ dense_result.intercept
end
