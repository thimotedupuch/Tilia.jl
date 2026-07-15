@testset "Feature agglomeration scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "feature_agglomeration_sklearn.toml"))
    X = reduce(vcat, permutedims.(fixture["case"]["X"]))
    fitted = fit(FeatureAgglomeration(n_clusters=2), X)
    expected = fixture["case"]["labels"]
    @test [fitted.labels[i] == fitted.labels[j] for i in 1:4, j in 1:4] ==
          [expected[i] == expected[j] for i in 1:4, j in 1:4]
end
