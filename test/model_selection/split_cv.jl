@testset "Splitting and cross-validation" begin
    folds = split(KFold(3), 8)
    @test length.(last.(folds)) == [3, 3, 2]
    @test sort!(reduce(vcat, last.(folds))) == collect(1:8)
    @test all(isempty(intersect(train, test)) for (train, test) in folds)

    shuffled_once = split(KFold(4; shuffle=true, seed=42), 12)
    shuffled_twice = split(KFold(4; shuffle=true, seed=42), 12)
    @test shuffled_once == shuffled_twice
    @test shuffled_once != split(KFold(4; shuffle=true, seed=43), 12)

    X = reshape(collect(1.0:24.0), 12, 2)
    y = repeat([:a, :b, :c], inner=4)
    first_split = train_test_split(X, y; test_size=6, seed=7, stratify=y)
    second_split = train_test_split(X, y; test_size=6, seed=7, stratify=y)
    @test first_split == second_split
    Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = first_split
    @test size(Xtrain, 1) == size(Xtest, 1) == 6
    @test sort(unique(ytrain)) == sort(unique(ytest)) == [:a, :b, :c]
    @test isempty(intersect(train_indices, test_indices))
    @test sort(vcat(train_indices, test_indices)) == collect(1:12)

    table = column_table((feature=collect(1:12), category=repeat([:a, :b, :c], 4)))
    table_train, table_test, _, _, table_train_indices, table_test_indices =
        train_test_split(table, y; test_size=3, seed=7)
    @test table_train isa ColumnTable
    @test table_test isa ColumnTable
    @test size(table_train) == (9, 2)
    @test size(table_test) == (3, 2)
    @test sort(vcat(table_train_indices, table_test_indices)) == collect(1:12)

    regression_X = [collect(1.0:12.0) ones(12)]
    regression_y = 2 .* regression_X[:, 1] .+ 3
    result = cross_validate(Chain(Standardize(), LinearRegression()), regression_X,
                            regression_y; cv=KFold(3))
    @test length(result.scores) == 3
    @test maximum(result.scores) < 1e-12
    @test [fold.stream_id for fold in result.fold_reports] ==
          ["root/cross_validation/fold/$index" for index in 1:3]
    @test all(isempty(intersect(train, test))
              for (train, test) in zip(result.train_indices, result.test_indices))
    # Each Standardize is fitted only on its fold's training rows.
    for (fold, train) in zip(result.fitted_models, result.train_indices)
        @test fold.fitted_nodes[1].means ≈ vec(mean(regression_X[train, :]; dims=1))
    end

    classification_X = [-3.0 0.0; -2.0 1.0; -1.0 0.0;
                         1.0 0.0; 2.0 -1.0; 3.0 0.0]
    classification_y = [:negative, :negative, :negative, :positive, :positive, :positive]
    classification = cross_validate(Chain(Standardize(), LogisticRegression(lambda=1.0)),
        classification_X, classification_y; cv=KFold(3; shuffle=true, seed=1))
    @test all(0 .<= classification.scores .<= 1)
end
