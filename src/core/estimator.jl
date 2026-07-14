"""Supertype for immutable, unfitted estimator specifications."""
abstract type AbstractEstimator end
abstract type AbstractTransformer <: AbstractEstimator end
abstract type AbstractPredictor <: AbstractEstimator end

input_contract(::AbstractEstimator) = (; rows_are_observations=true, accepts_missing=false)
output_schema(model::AbstractEstimator, schema) = schema
