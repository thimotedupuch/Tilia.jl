"""Supertype for differentiable numerical objectives, separate from fit control flow."""
abstract type AbstractObjective end

"""Data required by the analytic binary logistic objective."""
struct LogisticBatch{X,Y,W}
    design::X
    target::Y
    weights::W
    function LogisticBatch(design::X, target::Y, weights::W) where {X,Y,W}
        size(design, 1) == length(target) == length(weights) ||
            throw(DimensionMismatch("logistic design, target, and weights must have equal observation counts."))
        new{X,Y,W}(design, target, weights)
    end
end

"""Binary logistic loss plus an elementwise-masked L2 penalty."""
struct BinaryLogisticObjective{T,M} <: AbstractObjective
    lambda::T
    penalty_mask::M
    function BinaryLogisticObjective(lambda::T, penalty_mask::M) where {T<:Real,M<:AbstractVector}
        isfinite(lambda) && lambda >= zero(lambda) || throw(InvalidHyperparameterError(
            "binary logistic objective lambda must be finite and nonnegative."))
        new{T,M}(lambda, penalty_mask)
    end
end

function _check_objective_dimensions(objective::BinaryLogisticObjective, parameters, batch::LogisticBatch)
    size(batch.design, 2) == length(parameters) == length(objective.penalty_mask) ||
        throw(DimensionMismatch("logistic parameters, design columns, and penalty mask must agree."))
end

"""Evaluate an objective at `parameters` and `data`."""
function value(objective::BinaryLogisticObjective, parameters, batch::LogisticBatch)
    _check_objective_dimensions(objective, parameters, batch)
    logits = batch.design * parameters
    loss = zero(promote_type(eltype(logits), eltype(batch.weights)))
    for index in eachindex(batch.target)
        logit = logits[index]
        loss += batch.weights[index] *
                (max(logit, zero(logit)) - logit * batch.target[index] + log1p(exp(-abs(logit))))
    end
    loss + objective.lambda / 2 * sum(abs2, parameters .* objective.penalty_mask)
end

"""Write the analytic objective gradient into `destination`."""
function gradient!(destination, objective::BinaryLogisticObjective, parameters,
                   batch::LogisticBatch)
    _check_objective_dimensions(objective, parameters, batch)
    length(destination) == length(parameters) || throw(DimensionMismatch(
        "gradient destination and parameters must have equal lengths."))
    logits = batch.design * parameters
    probabilities = similar(logits)
    for index in eachindex(logits)
        logit = logits[index]
        probabilities[index] = logit >= zero(logit) ? inv(one(logit) + exp(-logit)) : begin
            exponential = exp(logit)
            exponential / (one(logit) + exponential)
        end
    end
    mul!(destination, batch.design', batch.weights .* (probabilities .- batch.target))
    destination .+= objective.lambda .* abs2.(objective.penalty_mask) .* parameters
    destination
end

"""Evaluate an objective and write its gradient in one protocol call."""
function value_and_gradient!(destination, objective::AbstractObjective, parameters, data)
    objective_value = value(objective, parameters, data)
    gradient!(destination, objective, parameters, data)
    objective_value, destination
end

"""Directional derivative of a scalar objective."""
function jvp(objective::AbstractObjective, parameters, tangent, data)
    length(parameters) == length(tangent) || throw(DimensionMismatch(
        "parameters and tangent must have equal lengths."))
    gradient = similar(parameters)
    gradient!(gradient, objective, parameters, data)
    dot(gradient, tangent)
end

"""Write a scalar-objective vector-Jacobian product into `destination`."""
function vjp!(destination, objective::AbstractObjective, parameters, cotangent, data)
    gradient!(destination, objective, parameters, data)
    destination .*= cotangent
    destination
end

function value(objective::AbstractObjective, parameters, data)
    throw(MethodError(value, (objective, parameters, data)))
end
function gradient!(destination, objective::AbstractObjective, parameters, data)
    throw(MethodError(gradient!, (destination, objective, parameters, data)))
end

"""Construct an AD-backed custom objective when DifferentiationInterface is loaded."""
function autodiff_objective(arguments...)
    throw(ArgumentError(
        "DifferentiationInterface is not loaded; add it to the active environment before constructing an AD-backed objective."))
end
