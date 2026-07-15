"""Full-covariance Gaussian mixture fitted by expectation-maximization."""
struct GaussianMixture <: AbstractEstimator
    n_components::Int
    n_init::Int
    max_iterations::Int
    tolerance::Float64
    regularization::Float64
    init::Symbol
    function GaussianMixture(; n_components::Integer=1, n_init::Integer=1,
                             max_iterations::Integer=100, tolerance::Real=1e-3,
                             regularization::Real=1e-6, init::Symbol=:kmeanspp)
        n_components > 0 || throw(InvalidHyperparameterError("GaussianMixture n_components must be positive."))
        n_init > 0 || throw(InvalidHyperparameterError("GaussianMixture n_init must be positive."))
        max_iterations > 0 || throw(InvalidHyperparameterError("GaussianMixture max_iterations must be positive."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "GaussianMixture tolerance must be finite and positive."))
        isfinite(regularization) && regularization > 0 || throw(InvalidHyperparameterError(
            "GaussianMixture regularization must be finite and positive."))
        init in (:kmeanspp, :random) || throw(InvalidHyperparameterError(
            "GaussianMixture init must be :kmeanspp or :random."))
        new(Int(n_components), Int(n_init), Int(max_iterations), Float64(tolerance),
            Float64(regularization), init)
    end
end

struct FittedGaussianMixture{M,T,R,S} <: AbstractFittedEstimator
    model::M
    means::Matrix{T}
    covariances::Vector{Matrix{T}}
    precisions::Vector{Matrix{T}}
    log_determinants::Vector{T}
    mixture_weights::Vector{T}
    lower_bound::T
    iterations::Int
    converged::Bool
    report::R
    schema::S
end

capabilities(::Type{<:GaussianMixture}) = (task=:clustering, sparse=false, missing=false,
    weights=false, partial_fit=false, probabilistic=true)

function _mixture_factors(covariances, regularization)
    precisions = Vector{Matrix{eltype(first(covariances))}}(undef, length(covariances))
    log_determinants = Vector{eltype(first(covariances))}(undef, length(covariances))
    for component in eachindex(covariances)
        precisions[component], log_determinants[component] =
            _regularized_precision(covariances[component], regularization, "GaussianMixture")
    end
    precisions, log_determinants
end

function _mixture_log_probabilities(data, means, precisions, log_determinants, mixture_weights)
    T = eltype(data)
    result = Matrix{T}(undef, size(data, 1), size(means, 1))
    constant_term = T(size(data, 2) * log(2pi))
    for component in axes(means, 1)
        centered = data .- view(means, component:component, :)
        quadratic = vec(sum((centered * precisions[component]) .* centered; dims=2))
        result[:, component] .= log(mixture_weights[component]) .-
            T(0.5) .* (constant_term + log_determinants[component] .+ quadratic)
    end
    result
end

function _mixture_expectation(data, means, covariances, mixture_weights, regularization)
    precisions, log_determinants = _mixture_factors(covariances, regularization)
    weighted_log_prob = _mixture_log_probabilities(
        data, means, precisions, log_determinants, mixture_weights)
    log_normalizers = Kernels.logsumexp(weighted_log_prob; dims=2)
    responsibilities = exp.(weighted_log_prob .- log_normalizers)
    mean(log_normalizers), responsibilities, precisions, log_determinants
end

function _initial_mixture(model, data, rng)
    initializer = KMeans(n_clusters=model.n_components, init=model.init, n_init=1,
                         max_iterations=20, tolerance=model.tolerance)
    means = _initial_centers(initializer, data, rng)
    n, p = size(data)
    centered = data .- mean(data; dims=1)
    global_covariance = transpose(centered) * centered / eltype(data)(n)
    covariances = [Matrix(global_covariance) for _ in 1:model.n_components]
    mixture_weights = fill(inv(eltype(data)(model.n_components)), model.n_components)
    means, covariances, mixture_weights
end

function _mixture_run(model, data, rng)
    means, covariances, mixture_weights = _initial_mixture(model, data, rng)
    T = eltype(data)
    history = T[]
    converged = false
    iterations = model.max_iterations
    responsibilities = zeros(T, size(data, 1), model.n_components)
    for iteration in 1:model.max_iterations
        lower_bound, responsibilities, _, _ = _mixture_expectation(
            data, means, covariances, mixture_weights, T(model.regularization))
        push!(history, lower_bound)
        if length(history) > 1 && abs(history[end] - history[end - 1]) <=
                T(model.tolerance) * max(abs(history[end - 1]), one(T))
            converged = true
            iterations = iteration
            break
        end
        component_weights = vec(sum(responsibilities; dims=1)) .+ eps(T)
        mixture_weights = component_weights ./ sum(component_weights)
        means = Matrix(transpose(transpose(data) * responsibilities ./ transpose(component_weights)))
        for component in 1:model.n_components
            centered = data .- view(means, component:component, :)
            covariances[component] = transpose(centered) *
                (centered .* view(responsibilities, :, component)) / component_weights[component]
        end
    end
    lower_bound, responsibilities, precisions, log_determinants = _mixture_expectation(
        data, means, covariances, mixture_weights, T(model.regularization))
    isempty(history) || (history[end] = lower_bound)
    (means=means, covariances=covariances, precisions=precisions,
     log_determinants=log_determinants, mixture_weights=mixture_weights,
     lower_bound=lower_bound, iterations=iterations, converged=converged, history=history)
end

function fit(model::GaussianMixture, X::AbstractMatrix; context=default_context())
    require_cpu(context, "GaussianMixture fitting")
    _validate_numeric_matrix(X, "GaussianMixture")
    n, p = size(X)
    p > 0 || throw(UnsupportedDataError("GaussianMixture requires at least one feature."))
    n >= model.n_components || throw(UnsupportedDataError(
        "GaussianMixture requires at least n_components observations."))
    data = Matrix{float(eltype(X))}(X)
    best = nothing
    for restart in 1:model.n_init
        restart_context = derive_context(context, :gaussian_mixture, :restart, restart)
        candidate = _mixture_run(model, data, restart_context.rng)
        (best === nothing || candidate.lower_bound > best.lower_bound) && (best = candidate)
    end
    warnings = best.converged ? String[] :
        ["GaussianMixture reached max_iterations without convergence."]
    details = (lower_bound=best.lower_bound, iterations=best.iterations,
               converged=best.converged, objective_history=best.history,
               n_components=model.n_components, n_init=model.n_init,
               covariance_type=:full)
    fit_report = FitReport(status=best.converged ? :success : :max_iterations,
        observations=n, features=p, backend=:cpu, warnings=warnings,
        details=details, context=context)
    FittedGaussianMixture(model, best.means, best.covariances, best.precisions,
        best.log_determinants, best.mixture_weights, best.lower_bound,
        best.iterations, best.converged, fit_report, infer_schema(X))
end

function predict_proba(fitted::FittedGaussianMixture, X::AbstractMatrix)
    _validate_numeric_matrix(X, "GaussianMixture")
    _validate_feature_count(fitted.schema, X, "GaussianMixture")
    data = Matrix{eltype(fitted.means)}(X)
    log_probabilities = _mixture_log_probabilities(data, fitted.means,
        fitted.precisions, fitted.log_determinants, fitted.mixture_weights)
    Kernels.softmax(log_probabilities; dims=2)
end

function predict(fitted::FittedGaussianMixture, X::AbstractMatrix)
    responsibilities = predict_proba(fitted, X)
    [argmax(view(responsibilities, row, :)) for row in axes(X, 1)]
end

report(fitted::FittedGaussianMixture) = fitted.report
