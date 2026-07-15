@testset "Capability-driven model discovery" begin
    catalog = model_catalog()
    types = map(entry -> entry.type, catalog)
    @test length(types) == length(unique(types))
    @test all(entry -> entry.capabilities.task != :unknown, catalog)

    classifiers = model_catalog(task=:classification)
    @test !isempty(classifiers)
    @test all(entry -> entry.capabilities.task == :classification, classifiers)
    @test LogisticRegression in map(entry -> entry.type, classifiers)
    @test DecisionTreeClassifier in map(entry -> entry.type, classifiers)

    sparse_estimators = model_catalog(sparse=true)
    @test Lasso in map(entry -> entry.type, sparse_estimators)
    @test SparseLogisticRegression in map(entry -> entry.type, sparse_estimators)
    @test all(entry -> entry.capabilities.sparse, sparse_estimators)

    probabilistic = model_catalog(task=:classification, probabilistic=true)
    @test all(entry -> entry.capabilities.task == :classification &&
                       entry.capabilities.probabilistic, probabilistic)
end
