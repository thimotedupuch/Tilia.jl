function _reactant_logistic_objective(X, target, weights, means, scales,
                                      coefficients, intercept, lambda)
    standardized = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)
    scores = standardized * coefficients .+ reshape(intercept, 1, :)
    losses = max.(scores, zero(eltype(scores))) .- scores .* target .+
             log1p.(exp.(-abs.(scores)))
    sum(losses .* weights) + sum(lambda) / 2 * sum(abs2, coefficients)
end
