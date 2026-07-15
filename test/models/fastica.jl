@testset "Fast independent component analysis" begin
    samples = 1_000
    time = collect(range(0.0, 8pi; length=samples))
    sources = hcat(sin.(time), sign.(sin.(1.7 .* time)))
    sources ./= std(sources; dims=1)
    mixing = [1.0 0.5; 0.3 1.2]
    X = sources * transpose(mixing)
    fitted = fit(FastICA(n_components=2, max_iterations=500, tolerance=1e-6), X;
                 context=FitContext(seed=13))
    recovered = transform(fitted, X)
    correlations = [abs(cor(view(sources, :, source), view(recovered, :, component)))
                    for source in 1:2, component in 1:2]
    @test all(maximum(view(correlations, source, :)) > 0.98 for source in 1:2)
    @test inverse_transform(fitted, recovered) ≈ X atol=1e-8
    @test fitted.converged
    @test fit(FastICA(), X; context=FitContext(seed=13)).unmixing ==
          fit(FastICA(), X; context=FitContext(seed=13)).unmixing
    fitted32 = fit(FastICA(max_iterations=300), Float32.(X))
    @test eltype(transform(fitted32, Float32.(X))) == Float32
    @test_throws Tilia.UnsupportedDataError fit(FastICA(n_components=3), X)
    @test_throws Tilia.InvalidHyperparameterError FastICA(n_components=0)
    @test_throws Tilia.InvalidHyperparameterError FastICA(tolerance=0)
end
