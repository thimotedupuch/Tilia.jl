using Tilia.Solvers

@testset "Least-squares solvers" begin
    X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
    y = [1.0, 3.0, 5.0]
    for solver in (:qr, :svd)
        result = least_squares(X, y; solver=solver)
        @test result.coefficients ≈ [1.0, 2.0] atol=1e-12
        @test result.residual_norm < 1e-12
        @test result.rank == 2
    end
    for solver in (:cholesky, :lsqr)
        result = least_squares(X, y; solver=solver)
        @test result.coefficients ≈ [1.0, 2.0] atol=1e-7
        @test result.residual_norm < 1e-7
    end

    rank_deficient = [1.0 2.0; 2.0 4.0; 3.0 6.0]
    result = least_squares(rank_deficient, [1.0, 2.0, 3.0]; solver=:svd)
    @test result.rank == 1
    @test rank_deficient * result.coefficients ≈ [1.0, 2.0, 3.0]
    @test result.coefficients ≈ [0.2, 0.4]

    ridge = ridge_least_squares(X, y, 2.0)
    @test X' * (X * ridge.coefficients - y) + 2ridge.coefficients ≈ zeros(2) atol=1e-12
    @test ridge_least_squares(X, y, 0.0).coefficients ≈ least_squares(X, y).coefficients
end
