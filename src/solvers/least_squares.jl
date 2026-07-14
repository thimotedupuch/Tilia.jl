struct LeastSquaresResult{C,T}
    coefficients::C
    rank::Int
    residual_norm::T
    solver::Symbol
end

function _svd_solution(X, y; tolerance=nothing)
    decomposition = svd(X; full=false)
    largest = isempty(decomposition.S) ? zero(real(eltype(X))) : maximum(decomposition.S)
    threshold = tolerance === nothing ? max(size(X)...) * eps(float(real(eltype(X)))) * largest : tolerance
    rank = count(value -> value > threshold, decomposition.S)
    inverse_values = map(value -> value > threshold ? inv(value) : zero(value), decomposition.S)
    coefficients = decomposition.V * (inverse_values .* (decomposition.U' * y))
    coefficients, rank
end

"""Solve an unregularized dense least-squares problem with QR or SVD."""
function least_squares(X::AbstractMatrix, y::AbstractVector; solver::Symbol=:qr, tolerance=nothing)
    size(X, 1) == length(y) || throw(DimensionMismatch("design rows and target length must agree."))
    coefficients, numerical_rank = if solver === :qr
        factorization = qr(X, ColumnNorm())
        factorization \ y, rank(factorization.R; atol=tolerance === nothing ? 0 : tolerance)
    elseif solver === :cholesky
        gram = Hermitian(transpose(X) * X)
        try
            cholesky(gram) \ (transpose(X) * y), size(X, 2)
        catch error
            error isa PosDefException || rethrow()
            throw(ArgumentError("Cholesky least squares requires a full-column-rank design."))
        end
    elseif solver === :svd
        _svd_solution(X, y; tolerance=tolerance)
    elseif solver === :lsqr
        result = lsqr(X, y; tolerance=tolerance === nothing ? 1e-8 : tolerance)
        result.parameters, size(X, 2)
    else
        throw(ArgumentError("least-squares solver must be :qr, :cholesky, :svd, or :lsqr."))
    end
    residual_norm = norm(X * coefficients - y)
    LeastSquaresResult(coefficients, numerical_rank, residual_norm, solver)
end

"""Solve `min ||Xβ-y||² + λ||β||²` using Cholesky or augmented SVD."""
function ridge_least_squares(X::AbstractMatrix, y::AbstractVector, lambda::Real;
                             solver::Symbol=:cholesky, tolerance=nothing)
    lambda >= zero(lambda) || throw(ArgumentError("ridge regularization must be nonnegative."))
    iszero(lambda) && return least_squares(X, y; solver=solver === :svd ? :svd : :qr,
                                            tolerance=tolerance)
    size(X, 1) == length(y) || throw(DimensionMismatch("design rows and target length must agree."))
    coefficients, numerical_rank = if solver === :cholesky
        gram = Symmetric(X' * X + lambda * I)
        cholesky(gram) \ (X' * y), size(X, 2)
    elseif solver === :svd
        decomposition = svd(X; full=false)
        shrinkage = decomposition.S ./ (abs2.(decomposition.S) .+ lambda)
        decomposition.V * (shrinkage .* (decomposition.U' * y)),
            count(value -> value > (tolerance === nothing ? max(size(X)...) * eps(float(eltype(X))) * maximum(decomposition.S; init=zero(eltype(decomposition.S))) : tolerance), decomposition.S)
    else
        throw(ArgumentError("ridge solver must be :cholesky or :svd."))
    end
    residual_norm = norm(X * coefficients - y)
    LeastSquaresResult(coefficients, numerical_rank, residual_norm, solver)
end
