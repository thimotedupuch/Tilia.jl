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
    metric_function = metric === :euclidean ? euclidean :
        metric === :squared_euclidean ? squared_euclidean :
        metric === :manhattan ? manhattan :
        metric === :cosine ? cosine_distance :
        throw(ArgumentError("unsupported distance metric: $metric."))
    result_type = float(promote_type(eltype(left), eltype(right)))
    result = Matrix{result_type}(undef, size(left, 1), size(right, 1))
    for j in axes(right, 1), i in axes(left, 1)
        result[i, j] = metric_function(view(left, i, :), view(right, j, :))
    end
    result
end
