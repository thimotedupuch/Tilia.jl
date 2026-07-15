@testset "DBSCAN scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "dbscan_sklearn.toml"))
    X = reduce(vcat, permutedims.(fixture["case"]["X"]))
    fitted = fit(DBSCAN(radius=0.25, min_neighbors=3), X)
    sklearn_labels = fixture["case"]["cluster_labels"]
    expected = [label < 0 ? 0 : label + 1 for label in sklearn_labels]
    @test fitted.labels == expected
    @test fitted.core_indices .- 1 == fixture["case"]["core_indices"]
end
