@testset "Nonnegative matrix factorization" begin
    W = Float64[1 0; 0.8 0.1; 0.1 0.9; 0 1; 0.5 0.5]
    H = Float64[2 1 0.2; 0.1 1 2]
    X = W * H
    fitted = fit(NMF(n_components=2, max_iterations=1_000, tolerance=1e-7), X;
                 context=FitContext(seed=12))
    @test fitted.reconstruction_error / norm(X) < 0.03
    @test all(>=(0), fitted.components)
    @test all(>=(0), fitted.embeddings)
    embedding = transform(fitted, X)
    @test size(embedding) == (5, 2)
    @test norm(X - inverse_transform(fitted, embedding)) / norm(X) < 0.04
    @test eltype(transform(fit(NMF(max_iterations=10), Float32.(X)), Float32.(X))) == Float32
    sparse_fitted = fit(NMF(n_components=2, max_iterations=100), sparse(X))
    @test size(transform(sparse_fitted, sparse(X))) == (5, 2)
    @test_throws Tilia.UnsupportedDataError fit(NMF(), X .- 1)
    @test_throws Tilia.UnsupportedDataError fit(NMF(n_components=4), X)
    @test_throws Tilia.InvalidHyperparameterError NMF(n_components=0)
end
