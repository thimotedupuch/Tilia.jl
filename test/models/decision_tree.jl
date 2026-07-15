@testset "Decision trees" begin
    X = [0.0 0; 0 1; 1 0; 1 1; 2 0; 2 1]
    y_class = [:low, :low, :middle, :middle, :high, :high]
    classifier = fit(DecisionTreeClassifier(), X, y_class)
    @test predict(classifier, X) == y_class
    probabilities = predict_proba(classifier, X)
    @test size(probabilities) == (6, 3)
    @test vec(sum(probabilities; dims=2)) ≈ ones(6)
    @test classifier.classes == [:high, :low, :middle]
    @test report(classifier).details.leaves == 3
    @test report(classifier).details.maximum_depth == 2
    @test sum(classifier.feature_importances) ≈ 1

    y_binary = ifelse.(X[:, 1] .<= 0, :left, :right)
    fast_binary = fit(DecisionTreeClassifier(max_depth=3), X, y_binary)
    general_binary = fit(DecisionTreeClassifier(max_depth=3), X, y_binary;
                         weights=ones(size(X, 1)))
    @test predict(fast_binary, X) == predict(general_binary, X)
    @test [(node.feature, node.threshold, node.left, node.right, node.is_leaf)
           for node in fast_binary.nodes] ==
          [(node.feature, node.threshold, node.left, node.right, node.is_leaf)
           for node in general_binary.nodes]

    shallow = fit(DecisionTreeClassifier(max_depth=1), X, y_class)
    @test report(shallow).details.maximum_depth == 1
    @test report(shallow).details.leaves == 2

    y_regression = [0.0, 0, 2, 2, 4, 4]
    regressor = fit(DecisionTreeRegressor(), X, y_regression)
    @test predict(regressor, X) == y_regression
    @test report(regressor).details.leaves == 3

    weighted_X = reshape([0.0, 1.0, 2.0], :, 1)
    weighted_y = [0.0, 10.0, 10.0]
    leaf = fit(DecisionTreeRegressor(max_depth=1, min_samples_split=4),
               weighted_X, weighted_y; weights=[8.0, 1.0, 1.0])
    @test predict(leaf, reshape([5.0], 1, 1)) == [2.0]

    X32 = Float32.(X)
    fitted32 = fit(DecisionTreeRegressor(), X32, Float32.(y_regression))
    @test eltype(predict(fitted32, X32)) == Float32
    @test capabilities(DecisionTreeClassifier()).probabilistic
    @test_throws Tilia.InvalidHyperparameterError DecisionTreeClassifier(criterion=:bad)
    @test_throws Tilia.InvalidHyperparameterError DecisionTreeRegressor(max_depth=0)
    @test_throws Tilia.SchemaMismatchError predict(classifier, ones(2, 3))
end
