function _reactant_probabilities(X, coefficients, intercept)
    scores = X * coefficients .+ reshape(intercept, 1, :)
    positive = one(eltype(scores)) ./ (one(eltype(scores)) .+ exp.(-scores))
    if size(coefficients, 2) == 1
        return cat(one(eltype(positive)) .- positive, positive; dims=2)
    end
    positive ./ sum(positive; dims=2)
end

function _reactant_regression(X, coefficients, intercept)
    X * coefficients .+ sum(intercept)
end

_reactant_transform_region(X, coefficients, offset) =
    X * coefficients .+ reshape(offset, 1, :)

function _reactant_clipped_transform_region(X, preclip_coefficients, preclip_offset,
                                            lower, upper, coefficients, offset)
    transformed = X * preclip_coefficients .+ reshape(preclip_offset, 1, :)
    clipped = clamp.(transformed, sum(lower), sum(upper))
    clipped * coefficients .+ reshape(offset, 1, :)
end

function _reactant_class_indices(X, coefficients, intercept)
    scores = X * coefficients .+ reshape(intercept, 1, :)
    if size(coefficients, 2) == 1
        one_score = one(eltype(scores))
        return ifelse.(vec(scores) .>= zero(eltype(scores)),
                       one_score + one_score, one_score)
    end
    class_count = size(coefficients, 2)
    indices = reshape(eltype(scores).(1:class_count), 1, :)
    sentinel = eltype(scores)(class_count + 1)
    candidates = ifelse.(scores .== maximum(scores; dims=2), indices, sentinel)
    vec(minimum(candidates; dims=2))
end

function _reactant_clipped_scores(X, preclip_coefficients, preclip_offset,
                                  lower, upper, coefficients, intercept)
    transformed = X * preclip_coefficients .+ reshape(preclip_offset, 1, :)
    clipped = clamp.(transformed, sum(lower), sum(upper))
    clipped * coefficients .+ reshape(intercept, 1, :)
end

function _reactant_clipped_probabilities(X, preclip_coefficients, preclip_offset,
                                         lower, upper, coefficients, intercept)
    scores = _reactant_clipped_scores(X, preclip_coefficients, preclip_offset,
                                      lower, upper, coefficients, intercept)
    positive = one(eltype(scores)) ./ (one(eltype(scores)) .+ exp.(-scores))
    size(coefficients, 2) == 1 &&
        return cat(one(eltype(positive)) .- positive, positive; dims=2)
    positive ./ sum(positive; dims=2)
end

function _reactant_clipped_regression(X, preclip_coefficients, preclip_offset,
                                      lower, upper, coefficients, intercept)
    transformed = X * preclip_coefficients .+ reshape(preclip_offset, 1, :)
    clipped = clamp.(transformed, sum(lower), sum(upper))
    clipped * coefficients .+ sum(intercept)
end

function _reactant_clipped_class_indices(X, preclip_coefficients, preclip_offset,
                                         lower, upper, coefficients, intercept)
    scores = _reactant_clipped_scores(X, preclip_coefficients, preclip_offset,
                                      lower, upper, coefficients, intercept)
    if size(coefficients, 2) == 1
        one_score = one(eltype(scores))
        return ifelse.(vec(scores) .>= zero(eltype(scores)),
                       one_score + one_score, one_score)
    end
    class_count = size(coefficients, 2)
    indices = reshape(eltype(scores).(1:class_count), 1, :)
    sentinel = eltype(scores)(class_count + 1)
    candidates = ifelse.(scores .== maximum(scores; dims=2), indices, sentinel)
    vec(minimum(candidates; dims=2))
end

function _reactant_imputed_scores(X, mask, fill_values, coefficients, intercept)
    filled = ifelse.(mask, reshape(fill_values, 1, :), X)
    filled * coefficients .+ reshape(intercept, 1, :)
end

function _reactant_imputed_probabilities(X, mask, fill_values, coefficients, intercept)
    scores = _reactant_imputed_scores(X, mask, fill_values, coefficients, intercept)
    positive = one(eltype(scores)) ./ (one(eltype(scores)) .+ exp.(-scores))
    size(coefficients, 2) == 1 &&
        return cat(one(eltype(positive)) .- positive, positive; dims=2)
    positive ./ sum(positive; dims=2)
end

function _reactant_imputed_regression(X, mask, fill_values, coefficients, intercept)
    filled = ifelse.(mask, reshape(fill_values, 1, :), X)
    filled * coefficients .+ sum(intercept)
end

function _reactant_imputed_class_indices(X, mask, fill_values, coefficients, intercept)
    scores = _reactant_imputed_scores(X, mask, fill_values, coefficients, intercept)
    if size(coefficients, 2) == 1
        one_score = one(eltype(scores))
        return ifelse.(vec(scores) .>= zero(eltype(scores)),
                       one_score + one_score, one_score)
    end
    class_count = size(coefficients, 2)
    indices = reshape(eltype(scores).(1:class_count), 1, :)
    sentinel = eltype(scores)(class_count + 1)
    candidates = ifelse.(scores .== maximum(scores; dims=2), indices, sentinel)
    vec(minimum(candidates; dims=2))
end
