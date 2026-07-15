@testset "Bernoulli restricted Boltzmann machine" begin
    patterns = [1.0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 1]
    X = repeat(patterns; outer=(20, 1))
    model = BernoulliRBM(n_components=3, learning_rate=0.1,
                         batch_size=8, n_iterations=20)
    fitted = fit(model, X)
    hidden = transform(fitted, X)
    reconstructed = inverse_transform(fitted, hidden)
    @test size(hidden) == (80, 3)
    @test size(reconstructed) == size(X)
    @test all(0 .< hidden .< 1)
    @test all(0 .< reconstructed .< 1)
    @test report(fitted).details.algorithm == :contrastive_divergence_1
    @test last(report(fitted).details.objective_history) <
          first(report(fitted).details.objective_history)
    repeated = fit(model, X)
    @test repeated.weights == fitted.weights
    @test transform(repeated, X) == hidden

    X32 = Float32.(X)
    fitted32 = fit(BernoulliRBM(n_components=2, n_iterations=1), X32)
    @test eltype(transform(fitted32, X32)) == Float32
    @test capabilities(BernoulliRBM()).task == :transformation
    @test !capabilities(BernoulliRBM()).probabilistic
    @test_throws Tilia.InvalidHyperparameterError BernoulliRBM(n_components=0)
    @test_throws Tilia.UnsupportedDataError fit(BernoulliRBM(), fill(2.0, 2, 2))
    @test_throws Tilia.SchemaMismatchError inverse_transform(fitted, ones(2, 4))
end
