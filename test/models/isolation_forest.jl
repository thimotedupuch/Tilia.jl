@testset "Isolation forest" begin
    inliers = hcat([0.05 * sin(index) for index in 1:80],
                   [0.05 * cos(index) for index in 1:80])
    anomalies = [5.0 5.0; -5.0 -5.0; 6.0 -6.0; -6.0 6.0]
    X = vcat(inliers, anomalies)
    model = IsolationForest(n_estimators=80, max_samples=0.75,
                            contamination=4 / 84, max_features=1.0)
    fitted = fit(model, X)
    scores = anomaly_score(fitted, X)
    @test mean(scores[end-3:end]) > maximum(scores[1:80])
    labels = predict(fitted, X)
    @test all(labels[end-3:end] .== -1)
    @test count(==(-1), labels) in 4:5
    @test length(fitted.trees) == 80
    @test report(fitted).details.sample_size == 63
    @test report(fitted).details.features_per_tree == 2

    repeated = fit(model, X)
    @test anomaly_score(repeated, X) == scores
    X32 = Float32.(X)
    fitted32 = fit(IsolationForest(n_estimators=3), X32)
    @test eltype(anomaly_score(fitted32, X32)) == Float32
    @test capabilities(IsolationForest()).task == :anomaly_detection
    @test_throws Tilia.InvalidHyperparameterError IsolationForest(n_estimators=0)
    @test_throws Tilia.InvalidHyperparameterError IsolationForest(contamination=0.8)
    @test_throws Tilia.SchemaMismatchError anomaly_score(fitted, ones(2, 3))
end
