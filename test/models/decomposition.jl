@testset "PCA and truncated SVD" begin
    X = [1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 9.0; 4.0 8.0 12.0]
    fitted = fit(PCA(n_components=1), X)
    scores = transform(fitted, X)
    @test size(scores) == (4, 1)
    @test size(fitted.components) == (3, 1)
    @test fitted.explained_variance_ratio[1] ≈ 1.0
    @test transpose(fitted.components) * fitted.components ≈ ones(1, 1)
    @test all(fitted.explained_variance .>= 0)
    @test inverse_transform(fitted, scores) ≈ X atol=1e-12
    @test report(fitted).details.centered
    @test fitted.components[argmax(abs.(fitted.components[:, 1])), 1] > 0

    whitened = fit(PCA(n_components=1, whiten=true), X)
    @test var(transform(whitened, X)[:, 1]; corrected=true) ≈ 1.0
    @test inverse_transform(whitened, transform(whitened, X)) ≈ X atol=1e-12

    general = [1.0 2 0; 0 1 3; 2 0 1; 4 1 2]
    truncated = fit(TruncatedSVD(n_components=2), general)
    @test truncated.mean == zeros(3)
    @test transform(truncated, general) ≈ general * truncated.components
    @test sum(truncated.explained_variance_ratio) <= 1 + eps()

    X32 = Float32.(general)
    @test eltype(transform(fit(PCA(n_components=2), X32), X32)) == Float32
    @test_throws Tilia.InvalidHyperparameterError PCA(n_components=0)
    @test_throws Tilia.InvalidHyperparameterError fit(PCA(n_components=4), general)
    @test_throws Tilia.SchemaMismatchError transform(fitted, ones(2, 2))
end
