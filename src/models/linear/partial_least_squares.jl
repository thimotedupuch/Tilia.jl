"""Partial least-squares regression with NIPALS latent components."""
struct PartialLeastSquaresRegression <: AbstractPredictor
    n_components::Int
    scale::Bool
    function PartialLeastSquaresRegression(; n_components::Integer=2,
                                           scale::Bool=true)
        n_components > 0 || throw(InvalidHyperparameterError(
            "PartialLeastSquaresRegression n_components must be positive."))
        new(Int(n_components), scale)
    end
end

struct FittedPartialLeastSquares{M,T,R,S} <: AbstractFittedEstimator
    model::M
    x_means::Vector{T}
    x_scales::Vector{T}
    x_weights::Matrix{T}
    x_loadings::Matrix{T}
    rotations::Matrix{T}
    y_loadings::Vector{T}
    coefficients::Vector{T}
    intercept::T
    report::R
    schema::S
end

capabilities(::Type{<:PartialLeastSquaresRegression}) = (task=:regression,
    sparse=false, missing=false, weights=false, partial_fit=false,
    probabilistic=false)

function fit(model::PartialLeastSquaresRegression, X::AbstractMatrix,
             y::AbstractVector; weights=nothing, context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "PartialLeastSquaresRegression fitting")
    _validate_regression_data(X, y, nothing, "PartialLeastSquaresRegression")
    n, p = size(X)
    model.n_components <= min(n - 1, p) || throw(UnsupportedDataError(
        "PartialLeastSquaresRegression n_components cannot exceed min(observations - 1, features)."))
    T = float(promote_type(eltype(X), eltype(y)))
    data, target = Matrix{T}(X), T.(y)
    x_means = vec(mean(data; dims=1))
    raw_scales = model.scale ? vec(std(data; dims=1, corrected=false)) : ones(T, p)
    x_scales = map(value -> iszero(value) ? one(T) : value, raw_scales)
    y_mean = mean(target)
    residual_X = (data .- transpose(x_means)) ./ transpose(x_scales)
    residual_y = target .- y_mean
    x_weights = Matrix{T}(undef, p, model.n_components)
    x_loadings = similar(x_weights)
    y_loadings = Vector{T}(undef, model.n_components)
    score_norms = Vector{T}(undef, model.n_components)
    for component in 1:model.n_components
        weight = transpose(residual_X) * residual_y
        weight_norm = norm(weight)
        weight_norm > eps(T) || throw(NumericalFailureError(
            "PartialLeastSquaresRegression exhausted target-correlated directions; reduce n_components."))
        weight ./= weight_norm
        score = residual_X * weight
        score_norm = dot(score, score)
        score_norm > eps(T) || throw(NumericalFailureError(
            "PartialLeastSquaresRegression produced a degenerate latent score."))
        loading = transpose(residual_X) * score / score_norm
        y_loading = dot(residual_y, score) / score_norm
        x_weights[:, component] .= weight
        x_loadings[:, component] .= loading
        y_loadings[component] = y_loading
        score_norms[component] = score_norm
        residual_X .-= score * transpose(loading)
        residual_y .-= y_loading .* score
    end
    cross_loadings = transpose(x_loadings) * x_weights
    rotations = Matrix(transpose(transpose(cross_loadings) \ transpose(x_weights)))
    scaled_coefficients = rotations * y_loadings
    coefficients = scaled_coefficients ./ x_scales
    intercept = y_mean - dot(x_means, coefficients)
    predictions = data * coefficients .+ intercept
    details = (components=model.n_components, scale=model.scale,
               solver=:nipals, latent_score_norms=score_norms,
               residual_sum_squares=sum(abs2, target .- predictions))
    FittedPartialLeastSquares(model, x_means, x_scales, x_weights,
        x_loadings, rotations, y_loadings, coefficients, intercept,
        FitReport(observations=n, features=p, details=details, context=context),
        with_target(infer_schema(X), y))
end

function transform(fitted::FittedPartialLeastSquares, X::AbstractMatrix)
    _validate_numeric_matrix(X, "PartialLeastSquaresRegression")
    _validate_feature_count(fitted.schema, X, "PartialLeastSquaresRegression")
    standardized = (X .- transpose(fitted.x_means)) ./ transpose(fitted.x_scales)
    standardized * fitted.rotations
end

function inverse_transform(fitted::FittedPartialLeastSquares,
                           scores::AbstractMatrix)
    size(scores, 2) == size(fitted.rotations, 2) || throw(SchemaMismatchError(
        "PartialLeastSquaresRegression score width must match n_components."))
    (scores * transpose(fitted.x_loadings)) .* transpose(fitted.x_scales) .+
        transpose(fitted.x_means)
end

function predict(fitted::FittedPartialLeastSquares, X::AbstractMatrix)
    _validate_numeric_matrix(X, "PartialLeastSquaresRegression")
    _validate_feature_count(fitted.schema, X, "PartialLeastSquaresRegression")
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedPartialLeastSquares) = fitted.report
