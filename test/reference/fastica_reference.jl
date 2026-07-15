@testset "FastICA independent-source reference" begin
    time = collect(range(0.0, 6pi; length=600))
    truth = hcat(sin.(time), tanh.(8 .* sin.(1.3 .* time)))
    X = truth * [1.0 0.4; -0.2 1.1]
    fitted = fit(FastICA(n_components=2, max_iterations=500, tolerance=1e-6), X;
                 context=FitContext(seed=81))
    recovered = transform(fitted, X)
    association = [abs(cor(view(truth, :, i), view(recovered, :, j)))
                   for i in 1:2, j in 1:2]
    @test minimum(maximum(association; dims=2)) > 0.97
    @test maximum(abs, transpose(recovered) * recovered / size(X, 1) - I) < 1e-6
end
