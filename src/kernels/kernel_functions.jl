function _kernel_gamma(gamma, feature_count, T)
    gamma === :scale && return inv(T(feature_count))
    gamma isa Real && isfinite(gamma) && gamma > 0 || throw(ArgumentError(
        "kernel gamma must be :scale or a finite positive number."))
    T(gamma)
end

"""Compute a linear, RBF, or polynomial Gram matrix between observation rows."""
function gram_matrix(left::AbstractMatrix, right::AbstractMatrix=left;
                     kernel::Symbol=:rbf, gamma=:scale, degree::Integer=3,
                     coef0::Real=1.0)
    size(left, 2) == size(right, 2) || throw(DimensionMismatch(
        "kernel inputs have different feature counts."))
    kernel in (:linear, :rbf, :polynomial) || throw(ArgumentError(
        "kernel must be :linear, :rbf, or :polynomial."))
    degree > 0 || throw(ArgumentError("polynomial degree must be positive."))
    T = float(promote_type(eltype(left), eltype(right)))
    left_data, right_data = Matrix{T}(left), Matrix{T}(right)
    kernel === :linear && return left_data * transpose(right_data)
    gamma_value = _kernel_gamma(gamma, size(left, 2), T)
    if kernel === :polynomial
        return (gamma_value .* (left_data * transpose(right_data)) .+ T(coef0)) .^ degree
    end
    squared = pairwise_distances(left_data, right_data; metric=:squared_euclidean)
    exp.(-gamma_value .* squared)
end
