"""Feature covariance with observations in rows."""
function covariance_matrix(X::AbstractMatrix; corrected::Bool=true)
    size(X, 1) > corrected || throw(ArgumentError("insufficient observations for covariance."))
    centered = X .- mean(X; dims=1)
    transpose(centered) * centered / (size(X, 1) - corrected)
end

"""Frequency-weighted feature covariance with observations in rows."""
function weighted_covariance(X::AbstractMatrix, weights::AbstractVector;
                             corrected::Bool=false)
    size(X, 1) == length(weights) || throw(DimensionMismatch(
        "covariance weights must match observation count."))
    all(weight -> isfinite(weight) && weight >= 0, weights) || throw(ArgumentError(
        "covariance weights must be finite and nonnegative."))
    total = sum(weights)
    denominator = total - corrected
    denominator > 0 || throw(ArgumentError("covariance weights have insufficient mass."))
    average = vec(sum(X .* weights; dims=1)) / total
    centered = X .- transpose(average)
    transpose(centered) * (centered .* weights) / denominator
end

"""Accumulate a contingency table and return `(matrix, row_levels, column_levels)`."""
function contingency_matrix(rows::AbstractVector, columns::AbstractVector;
                            row_levels=nothing, column_levels=nothing)
    length(rows) == length(columns) || throw(DimensionMismatch(
        "contingency inputs must have equal lengths."))
    row_levels === nothing && (row_levels = sort!(unique(rows)))
    column_levels === nothing && (column_levels = sort!(unique(columns)))
    row_lookup = Dict(value => index for (index, value) in enumerate(row_levels))
    column_lookup = Dict(value => index for (index, value) in enumerate(column_levels))
    counts = zeros(Int, length(row_levels), length(column_levels))
    for (row, column) in zip(rows, columns)
        haskey(row_lookup, row) && haskey(column_lookup, column) || throw(ArgumentError(
            "contingency level lists do not cover every observation."))
        counts[row_lookup[row], column_lookup[column]] += 1
    end
    counts, collect(row_levels), collect(column_levels)
end

"""Return sorted class labels and their counts."""
function class_counts(values::AbstractVector)
    levels = sort!(unique(values))
    levels, [count(==(level), values) for level in levels]
end

"""Count values in half-open bins, with the last right edge included."""
function histogram_counts(values::AbstractVector, edges::AbstractVector)
    length(edges) >= 2 && issorted(edges) && allunique(edges) || throw(ArgumentError(
        "histogram edges must be strictly increasing and contain at least two values."))
    counts = zeros(Int, length(edges) - 1)
    for value in values
        edges[1] <= value <= edges[end] || continue
        index = value == edges[end] ? length(counts) : searchsortedlast(edges, value)
        index > 0 && (counts[index] += 1)
    end
    counts
end
