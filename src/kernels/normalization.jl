"""Normalize every observation row using the L1, L2, or max norm."""
function normalize_rows(X::AbstractMatrix; norm::Symbol=:l2)
    magnitudes = if norm === :l2
        sqrt.(sum(abs2, X; dims=2))
    elseif norm === :l1
        sum(abs, X; dims=2)
    elseif norm === :max
        maximum(abs, X; dims=2)
    else
        throw(ArgumentError("norm must be :l1, :l2, or :max."))
    end
    safe_magnitudes = map(value -> iszero(value) ? one(value) : value, magnitudes)
    X ./ safe_magnitudes
end
