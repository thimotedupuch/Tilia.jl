"""Scale sparse matrix columns without changing its sparsity pattern."""
function scale_columns!(X::SparseMatrixCSC, scales::AbstractVector)
    size(X, 2) == length(scales) || throw(DimensionMismatch(
        "matrix has $(size(X, 2)) columns; received $(length(scales)) scales."))
    for column in axes(X, 2)
        for index in nzrange(X, column)
            X.nzval[index] *= scales[column]
        end
    end
    X
end

scale_columns(X::SparseMatrixCSC, scales::AbstractVector) = scale_columns!(copy(X), scales)

sparse_column_sums(X::SparseMatrixCSC) = vec(sum(X; dims=1))
sparse_dot(left::SparseVector, right::AbstractVector) = dot(left, right)
sparse_dot(left::AbstractVector, right::SparseVector) = dot(left, right)
sparse_matvec(X::SparseMatrixCSC, vector::AbstractVector) = X * vector

function center_sparse(X::SparseMatrixCSC, means::AbstractVector; allow_densify::Bool=false)
    length(means) == size(X, 2) || throw(DimensionMismatch("sparse means must match feature count."))
    all(iszero, means) && return copy(X)
    allow_densify || throw(ArgumentError(
        "sparse centering would densify the matrix; set allow_densify=true explicitly."))
    Matrix(X) .- transpose(means)
end
