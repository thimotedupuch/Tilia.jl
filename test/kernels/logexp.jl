using Tilia.Kernels

@testset "Stable transform kernels" begin
    @test logsumexp([1000.0, 1000.0]) ≈ 1000 + log(2)
    @test logsumexp(fill(-Inf, 2)) == -Inf
    logits = [1000.0 0.0 -1000.0; 1.0 2.0 3.0]
    probabilities = softmax(logits; dims=2)
    @test vec(sum(probabilities; dims=2)) ≈ ones(2)
    @test all(isfinite, probabilities)
    @test exp.(logsoftmax(logits; dims=2)) ≈ probabilities
    @test sigmoid(1000.0) == 1.0
    @test sigmoid(-1000.0) == 0.0
    @test sigmoid(0.0) == 0.5
    @test binary_cross_entropy([1000.0, -1000.0], [1.0, 0.0]) ≈ 0.0 atol=1e-12
    @test binary_cross_entropy([0.0], [1.0]; reduction=:none) ≈ [log(2)]
    @test_throws ArgumentError binary_cross_entropy([0.0], [2.0])

    normalized = normalize_rows([3.0 4.0; 0.0 0.0])
    @test normalized[1, :] ≈ [0.6, 0.8]
    @test normalized[2, :] == [0.0, 0.0]
    @test vec(sum(normalize_rows([1.0 3.0; 2.0 2.0]; norm=:l1); dims=2)) ≈ ones(2)
end
