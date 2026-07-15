"""Supertype for immutable, unfitted estimator specifications."""
abstract type AbstractEstimator end
abstract type AbstractTransformer <: AbstractEstimator end
abstract type AbstractPredictor <: AbstractEstimator end

function input_contract(model::AbstractEstimator)
    declared = capabilities(model)
    (; rows_are_observations=true, accepts_sparse=declared.sparse,
       accepts_missing=declared.missing, accepts_weights=declared.weights,
       task=declared.task)
end
output_schema(model::AbstractEstimator, schema) = schema
