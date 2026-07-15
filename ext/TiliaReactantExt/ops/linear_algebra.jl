function _reactant_probabilities(X, means, scales, coefficients, intercept)
    standardized = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)
    scores = standardized * coefficients .+ reshape(intercept, 1, :)
    positive = one(eltype(scores)) ./ (one(eltype(scores)) .+ exp.(-scores))
    if size(coefficients, 2) == 1
        return cat(one(eltype(positive)) .- positive, positive; dims=2)
    end
    positive ./ sum(positive; dims=2)
end
