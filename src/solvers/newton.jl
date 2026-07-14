struct NewtonResult{P,T}
    parameters::P
    objective_history::Vector{T}
    gradient_norm::T
    iterations::Int
    converged::Bool
end

"""
Iteratively reweighted least-squares solve for binary logistic regression.

For the canonical logit link, IRLS is algebraically the analytic Newton update;
this entry point shares the damped implementation and diagnostics.
"""
binary_logistic_irls(args...; kwargs...) = binary_logistic_newton(args...; kwargs...)

function _logistic_objective(design, target, observation_weights, parameters, lambda, penalty_mask)
    logits = design * parameters
    loss = zero(eltype(parameters))
    for index in eachindex(target)
        logit = logits[index]
        loss += observation_weights[index] *
                (max(logit, zero(logit)) - logit * target[index] + log1p(exp(-abs(logit))))
    end
    loss + lambda / 2 * sum(abs2, parameters .* penalty_mask)
end

function _sigmoid_vector!(probabilities, logits)
    for index in eachindex(logits)
        value = logits[index]
        probabilities[index] = value >= zero(value) ? inv(one(value) + exp(-value)) : begin
            exponential = exp(value)
            exponential / (one(value) + exponential)
        end
    end
    probabilities
end

"""Analytic damped-Newton solver for a binary logistic objective."""
function binary_logistic_newton(design::AbstractMatrix{T}, target::AbstractVector{T};
                                weights=ones(T, length(target)), lambda::T=zero(T),
                                penalty_mask=ones(T, size(design, 2)), max_iterations::Int=100,
                                tolerance::T=sqrt(eps(T))) where {T<:AbstractFloat}
    size(design, 1) == length(target) == length(weights) ||
        throw(DimensionMismatch("design, target, and weights observation counts must agree."))
    parameters = zeros(T, size(design, 2))
    probabilities = similar(target)
    objective_history = T[]
    gradient_norm = T(Inf)
    converged = false
    iterations = 0
    for iteration in 1:max_iterations
        iterations = iteration
        logits = design * parameters
        _sigmoid_vector!(probabilities, logits)
        residual = weights .* (probabilities .- target)
        gradient = design' * residual + lambda .* penalty_mask .* parameters
        gradient_norm = norm(gradient, Inf)
        objective = _logistic_objective(design, target, weights, parameters, lambda, penalty_mask)
        push!(objective_history, objective)
        if gradient_norm <= tolerance * (one(T) + norm(parameters, Inf))
            converged = true
            break
        end
        curvature = max.(weights .* probabilities .* (one(T) .- probabilities), eps(T))
        hessian = Symmetric(design' * (design .* curvature) +
                            Diagonal(lambda .* penalty_mask))
        step = try
            cholesky(hessian) \ gradient
        catch error
            if error isa PosDefException
                throw(LinearAlgebra.SingularException(0))
            end
            rethrow()
        end
        step_scale = one(T)
        directional_derivative = dot(gradient, step)
        accepted = false
        while step_scale >= T(2)^(-20)
            candidate = parameters .- step_scale .* step
            candidate_objective = _logistic_objective(
                design, target, weights, candidate, lambda, penalty_mask)
            if candidate_objective <= objective - T(1e-4) * step_scale * directional_derivative
                parameters = candidate
                accepted = true
                break
            end
            step_scale /= 2
        end
        accepted || break
    end
    NewtonResult(parameters, objective_history, gradient_norm, iterations, converged)
end
