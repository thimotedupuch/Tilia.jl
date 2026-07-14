"""Return declared estimator capabilities as a named tuple."""
capabilities(::Type{<:AbstractEstimator}) = (
    task=:unknown, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)
capabilities(model::AbstractEstimator) = capabilities(typeof(model))
