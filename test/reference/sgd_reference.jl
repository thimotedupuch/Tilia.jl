@testset "SGD closed-form numerical reference" begin
    X = hcat([i / 20 for i in 1:60],
             [sin(i / 4) for i in 1:60],
             [cos(i / 7) for i in 1:60])
    y = 1.5 .* X[:, 1] .- 0.75 .* X[:, 2] .+ 0.25 .* X[:, 3] .+ 0.4
    design = hcat(ones(size(X, 1)), X)
    reference = qr(design) \ y
    fitted = fit(SGDRegressor(learning_rate=0.2, l2=0.0, epochs=3_000,
                              batch_size=size(X, 1), shuffle=false), X, y)
    @test fitted.intercept ≈ reference[1] atol=5e-3
    @test fitted.coefficients ≈ reference[2:end] atol=5e-3

    labels = ifelse.(X[:, 1] .- X[:, 2] .> 1.0, :positive, :negative)
    classifier = fit(SGDClassifier(learning_rate=0.2, epochs=500,
        batch_size=15, shuffle=false), X, labels)
    @test accuracy_score(labels, predict(classifier, X)) >= 0.95
end
