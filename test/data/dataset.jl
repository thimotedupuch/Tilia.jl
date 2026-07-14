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
