@testset "Brute-force nearest neighbors" begin
    X = [0.0 0.0; 1.0 0.0; 0.0 2.0; 10.0 10.0]
    index = fit(NearestNeighbors(n_neighbors=2), X)
    distances, indices = kneighbors(index, [0.1 0.0])
    @test indices == [1 2]
    @test distances ≈ [0.1 0.9]
    @test transform(index, [0.1 0.0]) == distances

    tied = fit(NearestNeighbors(n_neighbors=3), [-1.0 0.0; 1.0 0.0; 0.0 2.0])
    tied_distances, tied_indices = kneighbors(tied, [0.0 0.0])
    @test tied_indices == [1 2 3]
    @test tied_distances ≈ [1.0 1.0 2.0]
    squared = fit(NearestNeighbors(n_neighbors=2, metric=:squared_euclidean), X)
    @test first(kneighbors(squared, [0.1 0.0])) ≈ [0.01 0.81]

    many = reshape(collect(1.0:70.0), :, 1)
    many_index = fit(NearestNeighbors(n_neighbors=65), many)
    many_distances, many_indices = kneighbors(many_index, reshape([1.0], 1, 1);
                                               n_neighbors=65)
    @test many_indices == reshape(collect(1:65), 1, :)
    @test many_distances == reshape(collect(0.0:64.0), 1, :)

    y_class = [:left, :left, :left, :right]
    classifier = fit(KNeighborsClassifier(n_neighbors=3), X, y_class)
    @test predict(classifier, [0.2 0.1; 9.5 10.0]) == [:left, :left]
    nearest_classifier = fit(KNeighborsClassifier(n_neighbors=3, weights=:distance), X, y_class)
    @test predict(nearest_classifier, [10.0 10.0]) == [:right]
    probabilities = predict_proba(nearest_classifier, [0.0 0.0; 10.0 10.0])
    @test vec(sum(probabilities; dims=2)) ≈ ones(2)
    @test probabilities[1, 1] == 1
    @test probabilities[2, 2] == 1

    y_regression = [0.0, 1.0, 2.0, 20.0]
    regressor = fit(KNeighborsRegressor(n_neighbors=2), X, y_regression)
    @test predict(regressor, [0.0 0.0]) == [0.5]
    distance_regressor = fit(KNeighborsRegressor(n_neighbors=2, weights=:distance), X, y_regression)
    @test predict(distance_regressor, [0.0 0.0]) == [0.0]

    X32 = Float32.(X)
    fitted32 = fit(KNeighborsRegressor(n_neighbors=2), X32, Float32.(y_regression))
    @test eltype(predict(fitted32, X32)) == Float32
    @test capabilities(KNeighborsClassifier()).task == :classification
    @test capabilities(KNeighborsRegressor()).task == :regression
    @test report(index).details.metric == :euclidean

    @test_throws Tilia.InvalidHyperparameterError NearestNeighbors(n_neighbors=0)
    @test_throws Tilia.InvalidHyperparameterError KNeighborsClassifier(weights=:bad)
    @test_throws Tilia.InvalidHyperparameterError fit(NearestNeighbors(n_neighbors=5), X)
    @test_throws Tilia.SchemaMismatchError kneighbors(index, ones(1, 3))
end
