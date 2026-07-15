@testset "Numeric preprocessing scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "preprocessing_sklearn.toml"))
    @test fixture["source"] == Dict("package" => "scikit-learn", "version" => "1.9.0")
    atol = fixture["tolerance"]["absolute"]
    rtol = fixture["tolerance"]["relative"]
    X = reduce(vcat, permutedims.(fixture["input"]["X"]))
    query = reduce(vcat, permutedims.(fixture["input"]["query"]))

    minimum = fit(MinMaxScale(feature_range=(-1.0, 1.0)), X)
    @test minimum.minima ≈ fixture["minmax"]["data_min"] atol=atol rtol=rtol
    @test minimum.ranges ≈ fixture["minmax"]["data_range"] atol=atol rtol=rtol
    @test transform(minimum, query) ≈
          reduce(vcat, permutedims.(fixture["minmax"]["query"])) atol=atol rtol=rtol

    normalized = transform(fit(Normalize(), X), query)
    @test normalized ≈ reduce(vcat,
        permutedims.(fixture["normalize"]["query"])) atol=atol rtol=rtol

    robust = fit(RobustScale(), X)
    @test robust.medians ≈ fixture["robust"]["center"] atol=atol rtol=rtol
    @test robust.scales ≈ fixture["robust"]["scale"] atol=atol rtol=rtol
    @test transform(robust, query) ≈ reduce(vcat,
        permutedims.(fixture["robust"]["query"])) atol=atol rtol=rtol

    polynomial = fit(PolynomialFeatures(degree=2), X)
    expected = reduce(vcat, permutedims.(fixture["polynomial"]["training"]))
    @test transform(polynomial, X) ≈ expected atol=atol rtol=rtol
    powers = [count(==(feature), term) for term in polynomial.terms, feature in 1:size(X, 2)]
    @test powers == reduce(vcat, permutedims.(fixture["polynomial"]["powers"]))
end
