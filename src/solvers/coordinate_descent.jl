struct CoordinateDescentResult{C,T,H}
    coefficients::C
    intercept::T
    objective_history::H
    iterations::Int
    converged::Bool
    maximum_update::T
end

_soft_threshold(value, threshold) = sign(value) * max(abs(value) - threshold, zero(value))

"""Solve an elastic-net least-squares objective by cyclic coordinate descent."""
function elastic_net_coordinate_descent(X::AbstractMatrix, y::AbstractVector;
        l1_penalty::Real, l2_penalty::Real, fit_intercept::Bool=true,
        weights=nothing, max_iterations::Integer=1_000, tolerance::Real=1e-6)
    size(X, 1) == length(y) || throw(DimensionMismatch("design rows and target length must agree."))
    l1_penalty >= 0 && l2_penalty >= 0 || throw(ArgumentError("penalties must be nonnegative."))
    T = float(promote_type(eltype(X), eltype(y), typeof(l1_penalty), typeof(l2_penalty),
                           weights === nothing ? eltype(y) : eltype(weights)))
    target = T.(y)
    observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
    total_weight = sum(observation_weights)
    coefficients = zeros(T, size(X, 2))
    intercept = fit_intercept ? sum(observation_weights .* target) / total_weight : zero(T)
    residual = target .- intercept
    history = T[]
    maximum_update = T(Inf)
    converged = false
    iterations = Int(max_iterations)
    for iteration in 1:max_iterations
        maximum_update = zero(T)
        for feature in axes(X, 2)
            column = view(X, :, feature)
            previous = coefficients[feature]
            residual .+= column .* previous
            denominator = sum(observation_weights .* abs2.(column)) / total_weight + T(l2_penalty)
            correlation = sum(observation_weights .* column .* residual) / total_weight
            updated = iszero(denominator) ? zero(T) :
                _soft_threshold(correlation, T(l1_penalty)) / denominator
            coefficients[feature] = updated
            residual .-= column .* updated
            maximum_update = max(maximum_update, abs(updated - previous))
        end
        if fit_intercept
            intercept_update = sum(observation_weights .* residual) / total_weight
            intercept += intercept_update
            residual .-= intercept_update
            maximum_update = max(maximum_update, abs(intercept_update))
        end
        objective = sum(observation_weights .* abs2.(residual)) / (T(2) * total_weight) +
                    T(l1_penalty) * sum(abs, coefficients) +
                    T(l2_penalty) * sum(abs2, coefficients) / T(2)
        push!(history, objective)
        if maximum_update <= T(tolerance)
            converged = true
            iterations = iteration
            break
        end
    end
    CoordinateDescentResult(coefficients, intercept, history, iterations,
                            converged, maximum_update)
end
