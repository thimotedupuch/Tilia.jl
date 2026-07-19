"""
Robust linear regression models, including Quantile, Huber, RANSAC, and Theil-Sen regression.
"""

struct QuantileRegression{T<:Real} <: AbstractPredictor
    quantile::T
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    epsilon::T
    function QuantileRegression(; quantile::Real=0.5, lambda::Real=0.0,
                                fit_intercept::Bool=true, max_iterations::Integer=1000,
                                tolerance::Real=1e-6, epsilon::Real=1e-4)
        0 < quantile < 1 || throw(InvalidHyperparameterError("QuantileRegression quantile must be in (0, 1); received $quantile."))
        lambda >= 0 || throw(InvalidHyperparameterError("QuantileRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("QuantileRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("QuantileRegression tolerance must be positive; received $tolerance."))
        epsilon > 0 || throw(InvalidHyperparameterError("QuantileRegression epsilon must be positive; received $epsilon."))
        T = promote_type(typeof(quantile), typeof(lambda), typeof(tolerance), typeof(epsilon))
        new{T}(T(quantile), T(lambda), fit_intercept, Int(max_iterations), T(tolerance), T(epsilon))
    end
end

struct FittedQuantileRegressor{M,C,I,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    report::R
    schema::S
end

capabilities(::Type{<:QuantileRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function quantile_objective(θ, X, y, weights, quantile, lambda, fit_intercept, epsilon)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[end] : zero(T)

    r = y .- (X * beta .+ b)
    loss = zero(T)
    for i in 1:n
        w = weights === nothing ? one(T) : T(weights[i])
        ri = r[i]
        abs_ri = abs(ri)
        h = abs_ri <= epsilon ? 0.5 * ri^2 / epsilon : abs_ri - 0.5 * epsilon
        loss += w * (0.5 * h + (quantile - 0.5) * ri)
    end
    reg = 0.5 * T(lambda) * sum(abs2, beta)
    return loss + reg
end

function quantile_gradient!(grad, θ, X, y, weights, quantile, lambda, fit_intercept, epsilon)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[end] : zero(T)

    r = y .- (X * beta .+ b)
    g = similar(r)
    for i in 1:n
        w = weights === nothing ? one(T) : T(weights[i])
        ri = r[i]
        abs_ri = abs(ri)
        h_prime = abs_ri <= epsilon ? ri / epsilon : sign(ri)
        g[i] = -w * (0.5 * h_prime + (quantile - 0.5))
    end

    grad_beta = view(grad, 1:d)
    mul!(grad_beta, transpose(X), g)
    grad_beta .+= T(lambda) .* beta

    if fit_intercept
        grad[end] = sum(g)
    end
    return grad
end

function fit(model::QuantileRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    model_name = "QuantileRegression"
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_regression_data(X, y, weights, model_name)

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)
    y_vec = Vector{T}(y)

    n_samples, n_features = size(X_mat)
    initial = zeros(T, model.fit_intercept ? n_features + 1 : n_features)

    if model.fit_intercept
        initial[end] = median(y_vec)
    end

    obj = θ -> quantile_objective(θ, X_mat, y_vec, weights, T(model.quantile), T(model.lambda), model.fit_intercept, T(model.epsilon))
    grad! = (g, θ) -> quantile_gradient!(g, θ, X_mat, y_vec, weights, T(model.quantile), T(model.lambda), model.fit_intercept, T(model.epsilon))

    result = Solvers.lbfgs(obj, grad!, initial;
                           tolerance=effective_tolerance(context, model.tolerance),
                           max_iterations=effective_max_iterations(context, model.max_iterations))

    coefficients = result.parameters[1:n_features]
    intercept = model.fit_intercept ? result.parameters[end] : zero(T)

    details = (solver=:lbfgs, objective_history=result.objective_history,
               iterations=result.iterations, converged=result.converged)

    fit_report = FitReport(status=result.converged ? :success : :max_iterations,
                           observations=n_samples, features=n_features,
                           backend=:cpu, details=details, context=context)

    FittedQuantileRegressor(model, coefficients, intercept, fit_report,
                            with_target(infer_schema(X), y))
end

function predict(fitted::FittedQuantileRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "QuantileRegression")
    _validate_feature_count(fitted.schema, X, "QuantileRegression")
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedQuantileRegressor) = fitted.report

# ----------------- Huber Regression -----------------

struct HuberRegression{T<:Real} <: AbstractPredictor
    epsilon::T
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    function HuberRegression(; epsilon::Real=1.35, lambda::Real=1.0,
                             fit_intercept::Bool=true, max_iterations::Integer=1000,
                             tolerance::Real=1e-6)
        epsilon > 1.0 || throw(InvalidHyperparameterError("HuberRegression epsilon must be > 1.0; received $epsilon."))
        lambda >= 0 || throw(InvalidHyperparameterError("HuberRegression lambda must be nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError("HuberRegression max_iterations must be positive; received $max_iterations."))
        tolerance > 0 || throw(InvalidHyperparameterError("HuberRegression tolerance must be positive; received $tolerance."))
        T = promote_type(typeof(epsilon), typeof(lambda), typeof(tolerance))
        new{T}(T(epsilon), T(lambda), fit_intercept, Int(max_iterations), T(tolerance))
    end
end

struct FittedHuberRegressor{M,C,I,S,R,Sch} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    scale::S
    report::R
    schema::Sch
end

capabilities(::Type{<:HuberRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)

function huber_objective(θ, X, y, weights, epsilon, lambda, fit_intercept)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[d+1] : zero(T)
    log_sigma = θ[end]
    sigma = exp(log_sigma)

    r = y .- (X * beta .+ b)
    loss = zero(T)
    limit = epsilon * sigma
    for i in 1:n
        w = weights === nothing ? one(T) : T(weights[i])
        ri = r[i]
        abs_ri = abs(ri)
        if abs_ri <= limit
            loss += w * (epsilon * sigma + (ri^2) / sigma)
        else
            loss += w * (2.0 * epsilon * abs_ri + (epsilon - epsilon^2) * sigma)
        end
    end
    reg = 0.5 * T(lambda) * sum(abs2, beta)
    return loss + reg
end

function huber_gradient!(grad, θ, X, y, weights, epsilon, lambda, fit_intercept)
    T = eltype(θ)
    n, d = size(X)
    beta = view(θ, 1:d)
    b = fit_intercept ? θ[d+1] : zero(T)
    log_sigma = θ[end]
    sigma = exp(log_sigma)

    r = y .- (X * beta .+ b)
    g = similar(r)
    d_sigma = zero(T)

    limit = epsilon * sigma
    for i in 1:n
        w = weights === nothing ? one(T) : T(weights[i])
        ri = r[i]
        abs_ri = abs(ri)
        if abs_ri <= limit
            g[i] = -w * (2.0 * ri / sigma)
            d_sigma += w * (epsilon - (ri / sigma)^2)
        else
            g[i] = -w * (2.0 * epsilon * sign(ri))
            d_sigma += w * (epsilon - epsilon^2)
        end
    end

    grad_beta = view(grad, 1:d)
    mul!(grad_beta, transpose(X), g)
    grad_beta .+= T(lambda) .* beta

    if fit_intercept
        grad[d+1] = sum(g)
    end

    grad[end] = d_sigma * sigma
    return grad
end

function fit(model::HuberRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    model_name = "HuberRegression"
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_regression_data(X, y, weights, model_name)

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)
    y_vec = Vector{T}(y)

    n_samples, n_features = size(X_mat)
    d = n_features
    initial = zeros(T, model.fit_intercept ? d + 2 : d + 1)
    if model.fit_intercept
        initial[d+1] = median(y_vec)
    end
    initial[end] = zero(T)

    obj = θ -> huber_objective(θ, X_mat, y_vec, weights, T(model.epsilon), T(model.lambda), model.fit_intercept)
    grad! = (g, θ) -> huber_gradient!(g, θ, X_mat, y_vec, weights, T(model.epsilon), T(model.lambda), model.fit_intercept)

    result = Solvers.lbfgs(obj, grad!, initial;
                           tolerance=effective_tolerance(context, model.tolerance),
                           max_iterations=effective_max_iterations(context, model.max_iterations))

    coefficients = result.parameters[1:d]
    intercept = model.fit_intercept ? result.parameters[d+1] : zero(T)
    scale = exp(result.parameters[end])

    details = (solver=:lbfgs, objective_history=result.objective_history,
               iterations=result.iterations, converged=result.converged,
               scale=scale)

    fit_report = FitReport(status=result.converged ? :success : :max_iterations,
                           observations=n_samples, features=n_features,
                           backend=:cpu, details=details, context=context)

    FittedHuberRegressor(model, coefficients, intercept, scale, fit_report,
                          with_target(infer_schema(X), y))
end

function predict(fitted::FittedHuberRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "HuberRegression")
    _validate_feature_count(fitted.schema, X, "HuberRegression")
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedHuberRegressor) = fitted.report

# ----------------- RANSAC Regression -----------------

struct RANSACRegression{E<:AbstractPredictor, T<:Real} <: AbstractPredictor
    base_estimator::E
    min_samples::Union{Int, Nothing}
    residual_threshold::Union{T, Nothing}
    max_trials::Int
    stop_probability::T
    loss::Symbol
    function RANSACRegression(; base_estimator::AbstractPredictor=LinearRegression(),
                              min_samples::Union{Integer, Nothing}=nothing,
                              residual_threshold::Union{Real, Nothing}=nothing,
                              max_trials::Integer=100, stop_probability::Real=0.99,
                              loss::Symbol=:absolute_error)
        max_trials > 0 || throw(InvalidHyperparameterError("RANSACRegression max_trials must be positive; received $max_trials."))
        stop_probability > 0 && stop_probability <= 1 || throw(InvalidHyperparameterError("RANSACRegression stop_probability must be in (0, 1]; received $stop_probability."))
        loss in (:absolute_error, :squared_error) || throw(InvalidHyperparameterError("RANSACRegression loss must be :absolute_error or :squared_error; received $loss."))
        E_type = typeof(base_estimator)
        T = promote_type(residual_threshold === nothing ? Float64 : typeof(residual_threshold), typeof(stop_probability))
        new{E_type, T}(base_estimator, min_samples === nothing ? nothing : Int(min_samples),
                        residual_threshold === nothing ? nothing : T(residual_threshold),
                        Int(max_trials), T(stop_probability), loss)
    end
end

struct FittedRANSACRegressor{M,F,I,R,S} <: AbstractFittedEstimator
    model::M
    fitted_base::F
    inlier_mask::I
    report::R
    schema::S
end

capabilities(::Type{<:RANSACRegression}) = (
    task=:regression, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=false,
)
capabilities(model::RANSACRegression) = merge(
    capabilities(typeof(model)),
    (weights=capabilities(model.base_estimator).weights,),
)

function _ransac_required_trials(stop_probability, inlier_ratio, min_samples, max_trials)
    inlier_ratio >= 1 && return 1
    stop_probability >= 1 && return max_trials
    clean_sample_probability = inlier_ratio^min_samples
    clean_sample_probability <= 0 && return max_trials
    required = ceil(Int, log1p(-stop_probability) / log1p(-clean_sample_probability))
    clamp(required, 1, max_trials)
end

function fit(model::RANSACRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    model_name = "RANSACRegression"
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_regression_data(X, y, weights, model_name)

    n_samples, n_features = size(X)
    min_samples_val = model.min_samples === nothing ? n_features + 1 : model.min_samples
    min_samples_val = clamp(min_samples_val, 1, n_samples)

    threshold_val = model.residual_threshold
    if threshold_val === nothing
        mad = median(abs.(y .- median(y)))
        threshold_val = max(mad, 1e-6)
    end

    best_inlier_mask = BitVector(undef, n_samples)
    best_num_inliers = 0
    trial_limit = model.max_trials
    trials_attempted = 0

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)
    y_vec = Vector{T}(y)

    for trial in 1:model.max_trials
        trial > trial_limit && break
        trials_attempted = trial
        sample_indices = randperm(context.rng, n_samples)[1:min_samples_val]
        sample_weights = weights === nothing ? nothing : weights[sample_indices]

        fitted_base = try
            fit(model.base_estimator, X_mat[sample_indices, :], y_vec[sample_indices];
                weights=sample_weights, context=context)
        catch
            continue
        end

        preds = predict(fitted_base, X_mat)

        if model.loss === :absolute_error
            residuals = abs.(y_vec .- preds)
        else
            residuals = (y_vec .- preds).^2
        end

        inlier_mask = residuals .<= threshold_val
        num_inliers = sum(inlier_mask)

        if num_inliers > best_num_inliers
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask
            inlier_ratio = best_num_inliers / n_samples
            trial_limit = min(trial_limit, _ransac_required_trials(
                model.stop_probability, inlier_ratio, min_samples_val,
                model.max_trials,
            ))
        end
    end

    if best_num_inliers == 0
        throw(NumericalFailureError("RANSAC was unable to find any valid consensus set."))
    end

    inlier_indices = findall(best_inlier_mask)
    final_weights = weights === nothing ? nothing : weights[inlier_indices]
    fitted_final = fit(model.base_estimator, X_mat[inlier_indices, :], y_vec[inlier_indices];
                       weights=final_weights, context=context)

    details = (best_trials_inliers=best_num_inliers, residual_threshold=threshold_val,
               inlier_mask=best_inlier_mask, trials_attempted=trials_attempted,
               trial_limit=trial_limit, stop_probability=model.stop_probability)

    fit_report = FitReport(status=:success, observations=n_samples, features=n_features,
                           backend=:cpu, details=details, context=context)

    FittedRANSACRegressor(model, fitted_final, best_inlier_mask, fit_report,
                          with_target(infer_schema(X), y))
end

function predict(fitted::FittedRANSACRegressor, X::AbstractMatrix)
    predict(fitted.fitted_base, X)
end

report(fitted::FittedRANSACRegressor) = fitted.report

# ----------------- Theil-Sen Regression -----------------

function spatial_median(vectors::AbstractMatrix; max_iterations=100, tolerance=1e-6)
    T = eltype(vectors)
    D, M = size(vectors)
    x = vec(median(vectors; dims=2))
    for iter in 1:max_iterations
        numerator = zeros(T, D)
        denominator = zero(T)
        for i in 1:M
            v = view(vectors, :, i)
            dist = norm(v - x)
            dist = max(dist, T(1e-10))
            numerator .+= v ./ dist
            denominator += inv(dist)
        end
        x_new = numerator ./ denominator
        if norm(x_new - x) < tolerance
            return x_new
        end
        x = x_new
    end
    return x
end

struct TheilSenRegression <: AbstractPredictor
    fit_intercept::Bool
    max_subpopulations::Int
    function TheilSenRegression(; fit_intercept::Bool=true, max_subpopulations::Integer=10000)
        max_subpopulations > 0 || throw(InvalidHyperparameterError("TheilSenRegression max_subpopulations must be positive; received $max_subpopulations."))
        new(fit_intercept, Int(max_subpopulations))
    end
end

struct FittedTheilSenRegressor{M,C,I,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    report::R
    schema::S
end

capabilities(::Type{<:TheilSenRegression}) = (
    task=:regression, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function fit(model::TheilSenRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    model_name = "TheilSenRegression"
    reject_unsupported_weights(model, weights)
    require_cpu(context, "$model_name fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError("$model_name sparse fitting is not supported yet."))
    _validate_regression_data(X, y, weights, model_name)

    n_samples, n_features = size(X)
    n_subsamples = model.fit_intercept ? n_features + 1 : n_features

    if n_samples < n_subsamples
        throw(UnsupportedDataError("$model_name requires at least $n_subsamples observations; received $n_samples."))
    end

    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    X_mat = Matrix{T}(X)
    y_vec = Vector{T}(y)

    total_combinations = try
        Base.binomial(Int128(n_samples), Int128(n_subsamples))
    catch
        typemax(Int128)
    end

    limit = min(total_combinations, model.max_subpopulations)

    sampled_sets = Set{Vector{Int}}()
    max_attempts = limit * 10
    attempts = 0
    while length(sampled_sets) < limit && attempts < max_attempts
        attempts += 1
        idx = sort!(randperm(context.rng, n_samples)[1:n_subsamples])
        push!(sampled_sets, idx)
    end

    betas = Matrix{T}(undef, n_features, length(sampled_sets))
    valid_count = 0

    for idx in sampled_sets
        X_sub = X_mat[idx, :]
        y_sub = y_vec[idx]

        design = model.fit_intercept ? hcat(X_sub, ones(T, n_subsamples)) : X_sub

        theta = try
            design \ y_sub
        catch
            continue
        end

        if any(!isfinite, theta)
            continue
        end

        valid_count += 1
        betas[:, valid_count] = view(theta, 1:n_features)
    end

    if valid_count == 0
        throw(NumericalFailureError("TheilSenRegression was unable to find any valid non-singular subsamples."))
    end

    actual_betas = view(betas, :, 1:valid_count)
    final_beta = spatial_median(actual_betas; max_iterations=200, tolerance=1e-6)

    final_intercept = model.fit_intercept ? median(y_vec .- X_mat * final_beta) : zero(T)

    details = (solver=:spatial_median, num_subpopulations=valid_count)
    fit_report = FitReport(status=:success, observations=n_samples, features=n_features,
                           backend=:cpu, details=details, context=context)

    FittedTheilSenRegressor(model, final_beta, final_intercept, fit_report,
                            with_target(infer_schema(X), y))
end

function predict(fitted::FittedTheilSenRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "TheilSenRegression")
    _validate_feature_count(fitted.schema, X, "TheilSenRegression")
    X * fitted.coefficients .+ fitted.intercept
end

report(fitted::FittedTheilSenRegressor) = fitted.report
