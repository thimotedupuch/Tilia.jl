function _reactant_logistic_objective(X, target, weights, means, scales,
                                      coefficients, intercept, lambda)
    standardized = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)
    scores = standardized * coefficients .+ reshape(intercept, 1, :)
    losses = max.(scores, zero(eltype(scores))) .- scores .* target .+
             log1p.(exp.(-abs.(scores)))
    sum(losses .* weights) + sum(lambda) / 2 * sum(abs2, coefficients)
end
function _reactant_standardize_statistics(X)
    count = eltype(X)(size(X, 1))
    means = sum(X; dims=1) ./ count
    centered = X .- means
    means, sum(abs2, centered; dims=1)
end

function _reactant_weighted_regression_statistics(X, target, weights)
    weighted_X = X .* reshape(weights, :, 1)
    weighted_target = target .* weights
    (sum(weights), sum(weighted_X; dims=1), sum(weighted_target),
     transpose(X) * weighted_X, transpose(X) * weighted_target,
     sum(weights .* abs2.(target)))
end

function _reactant_weighted_ridge_fit(X, target, weights, lambda, penalty_matrix,
                                      ::Val{FIT_INTERCEPT}) where {FIT_INTERCEPT}
    T = eltype(X)
    total_weight = sum(weights)
    feature_mean = FIT_INTERCEPT ? vec(sum(X .* reshape(weights, :, 1); dims=1)) ./
                                   total_weight : zeros(T, size(X, 2))
    target_mean = FIT_INTERCEPT ? sum(target .* weights) / total_weight : zero(T)
    centered_X = FIT_INTERCEPT ? X .- reshape(feature_mean, 1, :) : X
    centered_target = FIT_INTERCEPT ? target .- target_mean : target
    weighted_X = centered_X .* reshape(weights, :, 1)
    gram = transpose(centered_X) * weighted_X
    cross = transpose(centered_X) * (centered_target .* weights)
    regularized = gram + sum(lambda) .* penalty_matrix
    factorization = cholesky(Symmetric(regularized); check=false)
    valid = all(isfinite.(factorization.factors))
    coefficients = factorization \ cross
    intercept = target_mean - dot(feature_mean, coefficients)
    residual = sqrt(sum(weights .* abs2.(X * coefficients .+ intercept .- target)))
    coefficients, Reactant.broadcast_to_size(intercept, (1,)),
    Reactant.broadcast_to_size(residual, (1,)), valid
end

_reactant_weighted_ridge_fit_intercept(X, target, weights, lambda, penalty_matrix) =
    _reactant_weighted_ridge_fit(X, target, weights, lambda, penalty_matrix, Val(true))
_reactant_weighted_ridge_fit_no_intercept(X, target, weights, lambda, penalty_matrix) =
    _reactant_weighted_ridge_fit(X, target, weights, lambda, penalty_matrix, Val(false))


function _reactant_logistic_objective_value(design, target, weights, lambda,
                                            penalty_mask, parameters)
    T = eltype(design)
    logits = design * parameters
    exponential = exp.(-abs.(logits))
    losses = max.(logits, zero(T)) .- logits .* target .+ log1p.(exponential)
    sum(losses .* weights) + sum(lambda) / T(2) *
        sum(abs2.(parameters .* penalty_mask))
end

function _reactant_armijo_search(design, target, weights, lambda, penalty_mask,
                                 parameters, step, objective,
                                 directional_derivative, enabled, step_scales)
    T = eltype(design)
    candidates = reshape(parameters, :, 1) .-
                 reshape(step, :, 1) .* reshape(step_scales, 1, :)
    logits = design * candidates
    exponential = exp.(-abs.(logits))
    losses = max.(logits, zero(T)) .-
             logits .* reshape(target, :, 1) .+ log1p.(exponential)
    objectives = vec(sum(losses .* reshape(weights, :, 1); dims=1)) .+
                 sum(lambda) / T(2) .* vec(sum(abs2.(
                     candidates .* reshape(penalty_mask, :, 1)); dims=1))
    acceptable = enabled .& (objectives .<= objective .- T(1e-4) .* step_scales .*
                             directional_derivative)
    selected_scale = maximum(ifelse.(acceptable, step_scales, zero(T)))
    parameters .- selected_scale .* step, any(acceptable)
end

function _reactant_newton_iteration(design, target, weights, lambda,
                                    penalty_mask, penalty_matrix, tolerance, step_scales,
                                    parameters, objective_history, final_objective,
                                    final_gradient_norm, iterations, converged,
                                    valid_factorization, active, iteration)
    T = eltype(design)
    logits = design * parameters
    exponential = exp.(-abs.(logits))
    probabilities = ifelse.(logits .>= zero(T),
        one(T) ./ (one(T) .+ exponential),
        exponential ./ (one(T) .+ exponential))
    residual = weights .* (probabilities .- target)
    gradient = transpose(design) * residual .+
               sum(lambda) .* penalty_mask .* parameters
    gradient_norm = maximum(abs.(gradient))
    objective = _reactant_logistic_objective_value(
        design, target, weights, lambda, penalty_mask, parameters)
    next_history = Reactant.Ops.dynamic_update_slice(
        objective_history, Reactant.broadcast_to_size(objective, (1,)), [iteration])
    next_objective = ifelse(active, objective, final_objective)
    next_gradient_norm = ifelse(active, gradient_norm, final_gradient_norm)
    next_iterations = ifelse(active, T(iteration), iterations)
    criterion = gradient_norm <= sum(tolerance) *
                (one(T) + maximum(abs.(parameters)))
    next_converged = converged | (active & criterion)
    curvature = max.(weights .* probabilities .* (one(T) .- probabilities), eps(T))
    hessian = transpose(design) * (design .* curvature) .+
              sum(lambda) .* penalty_matrix
    factorization = cholesky(Symmetric(hessian); check=false)
    factorization_valid = all(isfinite.(factorization.factors))
    next_valid = valid_factorization & (!active | criterion | factorization_valid)
    step = factorization \ gradient
    candidate, accepted = _reactant_armijo_search(
        design, target, weights, lambda, penalty_mask, parameters, step,
        objective, dot(gradient, step), active & !criterion, step_scales)
    update = active & !criterion & accepted & factorization_valid
    next_parameters = ifelse.(update, candidate, parameters)
    next_parameters, next_history, next_objective, next_gradient_norm, next_iterations,
    next_converged, next_valid, update
end

function _reactant_binary_logistic_newton(design, target, weights, lambda,
                                          penalty_mask, penalty_matrix, tolerance,
                                          step_scales,
                                          ::Val{MAX_ITERATIONS}) where {MAX_ITERATIONS}
    T = eltype(design)
    parameters = zeros(T, size(design, 2))
    objective_history = zeros(T, MAX_ITERATIONS)
    final_objective = T(Inf)
    final_gradient_norm = T(Inf)
    iterations = zero(T)
    converged = false
    valid_factorization = true
    active = true
    @trace for iteration in 1:MAX_ITERATIONS
        parameters, objective_history, final_objective, final_gradient_norm, iterations,
        converged, valid_factorization, active = _reactant_newton_iteration(
            design, target, weights, lambda, penalty_mask, penalty_matrix,
            tolerance, step_scales, parameters, objective_history,
            final_objective, final_gradient_norm,
            iterations, converged, valid_factorization, active, iteration)
    end
    parameters, objective_history, final_objective, final_gradient_norm, iterations,
    converged, valid_factorization
end
