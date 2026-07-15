@testset "Dataset and schema" begin
    X = Float32[1 2; 3 4]
    data = Dataset(X; target=Float32[1, 2])
    @test data.features === X
    @test length(data.schema.columns) == 2
    @test data.schema.columns[1].name == :x1
    @test data.schema.columns[1].physical_type == Float32
    @test fit(MeanRegressor(), data).mean == 1.5f0
    @test_throws Tilia.SchemaMismatchError Dataset(X; target=[1])
end

@testset "Target schema metadata" begin
    regression = Dataset(ones(Float32, 3, 2); target=Float32[1, 2, 3])
    @test regression.schema.target_name == :target
    @test regression.schema.target_logical_type == :continuous
    @test regression.schema.target_physical_type == Float32
    @test !regression.schema.target_allows_missing

    classification = Dataset((feature=[1, 2, 3],); target=[:a, :b, :a])
    @test classification.schema.target_logical_type == :categorical
    @test classification.schema.target_physical_type == Symbol
end

@testset "Numerical estimators accept native and external tables" begin
    source = (length=[1.0, 2.0, 3.0, 4.0], width=[2.0, 1.0, 4.0, 3.0])
    target = [1.0, 2.0, 3.0, 4.0]
    fitted = fit(RidgeRegression(lambda=0.1), source, target)
    @test predict(fitted, source) ≈ predict(fitted, hcat(source.length, source.width))
    @test [column.name for column in fitted.schema.columns] == [:length, :width]

    dataset = Dataset(source; target)
    @test dataset.features.schema === dataset.schema
    from_dataset = fit(RidgeRegression(lambda=0.1), dataset)
    @test predict(from_dataset, dataset.features) ≈ predict(fitted, source)
    @test predict(from_dataset, dataset) ≈ predict(fitted, source)

    transformed = fit(Standardize(), column_table(source))
    @test size(transform(transformed, source)) == (4, 2)
    pipeline = fit(Chain(Standardize(), RidgeRegression(lambda=0.1)), source, target)
    @test size(predict(pipeline, source)) == (4,)
    dataset_pipeline = fit(Chain(Standardize(), RidgeRegression(lambda=0.1)), dataset)
    @test predict(dataset_pipeline, dataset) ≈ predict(pipeline, source)
end
