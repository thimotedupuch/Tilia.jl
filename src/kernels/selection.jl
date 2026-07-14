"""Indices of the `k` largest or smallest values with stable index tie-breaking."""
function topk_indices(values::AbstractVector, k::Integer; largest::Bool=true)
    0 <= k <= length(values) || throw(ArgumentError("k must lie between zero and the input length."))
    k == 0 && return Int[]
    ordering = sortperm(eachindex(values); by=index ->
        largest ? (-values[index], index) : (values[index], index))
    ordering[1:k]
end

"""Linear-interpolated sample quantile for `probability` in `[0,1]`."""
function quantile_value(values::AbstractVector, probability::Real)
    isempty(values) && throw(ArgumentError("quantile input cannot be empty."))
    0 <= probability <= 1 || throw(ArgumentError("quantile probability must lie in [0,1]."))
    ordered = sort(values)
    position = one(float(probability)) + probability * (length(values) - 1)
    lower = floor(Int, position)
    upper = ceil(Int, position)
    fraction = position - lower
    (one(fraction) - fraction) * ordered[lower] + fraction * ordered[upper]
end

"""Ranks with explicit `:average`, `:min`, `:max`, or `:dense` tie handling."""
function rank_values(values::AbstractVector; ties::Symbol=:average)
    ties in (:average, :min, :max, :dense) || throw(ArgumentError(
        "ties must be :average, :min, :max, or :dense."))
    ordering = sortperm(values; alg=MergeSort)
    ranks = zeros(Float64, length(values))
    position = 1
    dense_rank = 1
    while position <= length(ordering)
        stop = position
        while stop < length(ordering) && values[ordering[stop + 1]] == values[ordering[position]]
            stop += 1
        end
        rank = ties === :average ? (position + stop) / 2 :
               ties === :min ? position : ties === :max ? stop : dense_rank
        for index in position:stop
            ranks[ordering[index]] = rank
        end
        position = stop + 1
        dense_rank += 1
    end
    ranks
end
