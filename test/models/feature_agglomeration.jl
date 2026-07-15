@testset "Feature agglomeration" begin
    x = collect(range(-2.0, 2.0; length=80))
    z = sin.(3 .* x)
    X = hcat(x, x .+ 0.001 .* cos.(x), z, z .+ 0.001 .* x)
    fitted = fit(FeatureAgglomeration(n_clusters=2), X)
    @test fitted.labels[1] == fitted.labels[2]
    @test fitted.labels[3] == fitted.labels[4]
    @test fitted.labels[1] != fitted.labels[3]
    reduced = transform(fitted, X)
    @test size(reduced) == (80, 2)
    reconstructed = inverse_transform(fitted, reduced)
    @test size(reconstructed) == size(X)
    @test reconstructed[:, 1] == reconstructed[:, 2]
    @test reconstructed[:, 3] == reconstructed[:, 4]
    @test report(fitted).details.original_features == 4
    @test eltype(transform(fit(FeatureAgglomeration(), Float32.(X)), Float32.(X))) == Float32
    @test_throws Tilia.UnsupportedDataError fit(FeatureAgglomeration(n_clusters=5), X)
    @test_throws Tilia.InvalidHyperparameterError FeatureAgglomeration(n_clusters=0)
    @test_throws Tilia.InvalidHyperparameterError FeatureAgglomeration(linkage=:ward)
end
