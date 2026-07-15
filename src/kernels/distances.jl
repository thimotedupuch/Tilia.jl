function _check_vectors(left, right)
    length(left) == length(right) || throw(DimensionMismatch(
        "vectors have lengths $(length(left)) and $(length(right))."))
end

"""Squared Euclidean distance between two vectors."""
function squared_euclidean(left::AbstractVector, right::AbstractVector)
    _check_vectors(left, right)
    sum(abs2, left .- right)
end

euclidean(left::AbstractVector, right::AbstractVector) = sqrt(squared_euclidean(left, right))

function manhattan(left::AbstractVector, right::AbstractVector)
    _check_vectors(left, right)
    sum(abs, left .- right)
end

function cosine_distance(left::AbstractVector, right::AbstractVector)
    _check_vectors(left, right)
    denominator = stable_norm(left) * stable_norm(right)
    iszero(denominator) && throw(ArgumentError("cosine distance is undefined for a zero vector."))
    one(float(denominator)) - dot(left, right) / denominator
end

"""Mahalanobis distance using a supplied inverse covariance matrix."""
function mahalanobis_distance(left::AbstractVector, right::AbstractVector,
                              precision::AbstractMatrix)
    _check_vectors(left, right)
    size(precision) == (length(left), length(left)) || throw(DimensionMismatch(
        "precision matrix shape must match vector length."))
    difference = left .- right
    squared = dot(difference, precision * difference)
    squared >= -sqrt(eps(float(eltype(precision)))) || throw(ArgumentError(
        "precision matrix produced a negative squared distance."))
    sqrt(max(squared, zero(squared)))
end

"""Pairwise distances between observation rows of `left` and `right`."""
function pairwise_distances(left::AbstractMatrix, right::AbstractMatrix=left; metric::Symbol=:euclidean)
    size(left, 2) == size(right, 2) || throw(DimensionMismatch(
        "matrices have $(size(left, 2)) and $(size(right, 2)) features."))
    result_type = float(promote_type(eltype(left), eltype(right)))
    if metric === :euclidean || metric === :squared_euclidean
        left_data = eltype(left) === result_type ? left : result_type.(left)
        right_data = eltype(right) === result_type ? right : result_type.(right)
        result = Matrix{result_type}(left_data * transpose(right_data))
        left_norms = vec(sum(abs2, left_data; dims=2))
        right_norms = vec(sum(abs2, right_data; dims=2))
        result .= max.(left_norms .+ transpose(right_norms) .- 2 .* result,
                       zero(result_type))
        metric === :euclidean && (result .= sqrt.(result))
        return result
    end
    metric_function = metric === :euclidean ? euclidean :
        metric === :squared_euclidean ? squared_euclidean :
        metric === :manhattan ? manhattan :
        metric === :cosine ? cosine_distance :
        throw(ArgumentError("unsupported distance metric: $metric."))
    result = Matrix{result_type}(undef, size(left, 1), size(right, 1))
    for j in axes(right, 1), i in axes(left, 1)
        result[i, j] = metric_function(view(left, i, :), view(right, j, :))
    end
    result
end

"""Compute pairwise distances in bounded row blocks.

Set `threaded=true` to schedule independent output blocks across Julia threads.
Results are deterministic because every task owns disjoint result rows.
"""
function pairwise_distance_blocks(left::AbstractMatrix, right::AbstractMatrix=left;
                                  metric::Symbol=:euclidean,
                                  block_size::Integer=1024,
                                  threaded::Bool=false)
    block_size > 0 || throw(ArgumentError("block_size must be positive."))
    size(left, 2) == size(right, 2) || throw(DimensionMismatch(
        "matrices have $(size(left, 2)) and $(size(right, 2)) features."))
    T = float(promote_type(eltype(left), eltype(right)))
    result = Matrix{T}(undef, size(left, 1), size(right, 1))
    starts = collect(1:block_size:size(left, 1))
    compute_block = function (block_index)
        first_row = starts[block_index]
        last_row = min(first_row + block_size - 1, size(left, 1))
        result[first_row:last_row, :] .= pairwise_distances(
            view(left, first_row:last_row, :), right; metric)
    end
    if threaded && length(starts) > 1
        Threads.@threads for block_index in eachindex(starts)
            compute_block(block_index)
        end
    else
        for block_index in eachindex(starts)
            compute_block(block_index)
        end
    end
    result
end
