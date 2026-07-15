@testset "NMF exact nonnegative-rank reference" begin
    X = Float64[2 1 0; 1 0.5 0; 0 0.5 1; 0 1 2]
    fitted = fit(NMF(n_components=2, max_iterations=2_000, tolerance=1e-8), X;
                 context=FitContext(seed=5))
    @test fitted.reconstruction_error / norm(X) < 1e-2
    @test all(diff(report(fitted).details.objective_history) .<= 1e-10)
end
