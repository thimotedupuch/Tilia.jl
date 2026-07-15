@testset "Native tables and mixed preprocessing" begin
    age = Union{Missing,Float64}[20, missing, 40, 50]
    city = Union{Missing,String}["Paris", "Lyon", missing, "Paris"]
    source = (age=age, city=city)
    table = column_table(source)

    @test table isa ColumnTable
    @test size(table) == (4, 2)
    @test table.names == (:age, :city)
    @test table.columns[2] isa CategoricalColumn
    @test table.columns[2].pool == ["Lyon", "Paris"]
    @test table.columns[2].codes == [2, 1, 0, 2]
    @test table.schema.columns[1].logical_type == :continuous
    @test table.schema.columns[1].allows_missing
    @test table.schema.columns[2].logical_type == :categorical
    @test table.schema.columns[2].levels == ["Lyon", "Paris"]
    @test !table.schema.columns[2].ordered
    @test table.schema.columns[2].unknown_policy == :error
    @test table.schema.columns[2].missing_policy == :allow
    @test table.schema.columns[2].code_type <: Integer
    @test isequal(Tables.columntable(table).age, age)

    imputer = fit(Impute(), table)
    imputed = transform(imputer, table)
    @test imputer.fill_values == (110 / 3, "Paris")
    @test imputed.columns[1] ≈ [20, 110 / 3, 40, 50]
    @test imputed.columns[2] == ["Paris", "Lyon", "Paris", "Paris"]
    @test all(!column.allows_missing for column in imputed.schema.columns)
    @test report(imputer).details.missing_counts == [1, 1]

    encoder = fit(OneHotEncode(output_type=Float32), imputed)
    encoded = transform(encoder, imputed)
    @test encoded isa Matrix{Float32}
    @test size(encoded) == (4, 3)
    @test encoded[:, 1] ≈ Float32[20, 110 / 3, 40, 50]
    @test encoded[:, 2:3] == Float32[0 1; 1 0; 0 1; 0 1]
    @test [column.name for column in encoder.schema.columns] ==
          [:age, :city__Lyon, :city__Paris]
    @test encoder.schema.columns[2].provenance == [:city]
    @test encoder.schema.columns[2].generated_name == :city__Lyon

    unknown = (age=[30.0], city=["Marseille"])
    @test_throws Tilia.SchemaMismatchError transform(encoder, unknown)
    ignoring = fit(OneHotEncode(handle_unknown=:ignore), imputed)
    @test transform(ignoring, unknown) == [30.0 0.0 0.0]

    constant = fit(Impute(strategy=:constant, fill_value=0),
                   Union{Missing,Float64}[1 missing; missing 4])
    @test transform(constant, Union{Missing,Float64}[missing 2; 3 missing]) == [0.0 2.0; 3.0 0.0]
    @test_throws Tilia.UnsupportedDataError fit(Impute(),
        (value=Union{Missing,Float64}[missing, missing],))
end

@testset "Mixed table end-to-end workflow" begin
    source = (
        age=Union{Missing,Float64}[20, 22, missing, 45, 48, 52, 19, 50, 25, 42, 21, 47],
        city=Union{Missing,String}["A", "A", "B", "B", "C", missing,
                                      "A", "C", "B", "B", "A", "C"],
    )
    target = [:low, :low, :low, :high, :high, :high,
              :low, :high, :low, :high, :low, :high]
    pipeline = Chain(Impute(), OneHotEncode(handle_unknown=:ignore),
                     Standardize(), LogisticRegression(lambda=1.0))
    fitted = fit(pipeline, source, target)
    probabilities = predict_proba(fitted, source)
    @test size(probabilities) == (12, 2)
    @test accuracy_score(target, predict(fitted, source)) >= 11 / 12
    @test report(fitted).details.nodes == 4

    result = cross_validate(pipeline, source, target;
                            cv=KFold(3; shuffle=true, seed=9))
    @test length(result.scores) == 3
    @test all(0 .<= result.scores .<= 1)
    # Every fold owns an independently fitted imputation state.
    @test length(unique([fold.fitted_nodes[1].fill_values[1]
                         for fold in result.fitted_models])) > 1

    mktempdir() do directory
        path = joinpath(directory, "mixed_pipeline")
        @test save_model(path, fitted) == path
        loaded = load_model(path)
        @test predict(loaded, source) == predict(fitted, source)
        @test predict_proba(loaded, source) ≈ predict_proba(fitted, source)
        @test report(loaded).details.loaded
        @test report(loaded).root_seed == report(fitted).root_seed
        @test report(loaded).stream_id == report(fitted).stream_id
        @test isfile(joinpath(path, "manifest.toml"))
        @test isfile(joinpath(path, "specification.toml"))
        @test isfile(joinpath(path, "schema.toml"))
        @test isfile(joinpath(path, "report.toml"))
        @test !isempty(readdir(joinpath(path, "arrays")))

        open(joinpath(path, "arrays", first(readdir(joinpath(path, "arrays")))), "a") do io
            write(io, UInt8(0xff))
        end
        @test_throws Tilia.PersistenceVersionError load_model(path)
    end
end
