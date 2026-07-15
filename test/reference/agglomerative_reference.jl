@testset "Agglomerative scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "agglomerative_sklearn.toml"))
    X = reduce(vcat, permutedims.(fixture["case"]["X"]))
    fitted = fit(AgglomerativeClustering(n_clusters=2, linkage=:average), X)
    sklearn = fixture["case"]["labels"]
    expected_pairs = [sklearn[i] == sklearn[j] for i in eachindex(sklearn),
                      j in eachindex(sklearn)]
    actual_pairs = [fitted.labels[i] == fitted.labels[j]
                    for i in eachindex(fitted.labels), j in eachindex(fitted.labels)]
    @test actual_pairs == expected_pairs
end
