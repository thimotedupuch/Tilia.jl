"""Return declared estimator capabilities as a named tuple."""
capabilities(::Type{<:AbstractEstimator}) = (
    task=:unknown, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)
capabilities(model::AbstractEstimator) = capabilities(typeof(model))

function reject_unsupported_weights(model::AbstractEstimator, weights)
    weights === nothing && return nothing
    capabilities(model).weights && return nothing
    throw(UnsupportedDataError(
        "$(nameof(typeof(model))) does not support observation weights; omit weights or choose an estimator with capabilities(model).weights == true."))
end
