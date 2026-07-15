@testset "Agglomerative clustering" begin
    X = Float64[0 0; 0.1 0; 0 0.1; 4 4; 4.1 4; 4 4.1]
    for linkage in (:single, :complete, :average)
        fitted = fit(AgglomerativeClustering(n_clusters=2, linkage=linkage), X)
        @test fitted.labels == [1, 1, 1, 2, 2, 2]
        @test size(fitted.children) == (4, 2)
        @test issorted(fitted.merge_distances)
        @test predict(fitted, [0.05 0.05; 4.05 4.05]) == [1, 2]
    end
    singleton = fit(AgglomerativeClustering(n_clusters=6), X)
    @test singleton.labels == collect(1:6)
    @test isempty(singleton.merge_distances)
    @test_throws Tilia.InvalidHyperparameterError AgglomerativeClustering(n_clusters=0)
    @test_throws Tilia.InvalidHyperparameterError AgglomerativeClustering(linkage=:ward)
    @test_throws Tilia.UnsupportedDataError fit(
        AgglomerativeClustering(n_clusters=7), X)
end
