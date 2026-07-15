"""Fast independent-component analysis with symmetric fixed-point updates."""
struct FastICA <: AbstractTransformer
    n_components::Int
    max_iterations::Int
    tolerance::Float64
    function FastICA(; n_components::Integer=2, max_iterations::Integer=200,
                     tolerance::Real=1e-4)
        n_components > 0 || throw(InvalidHyperparameterError(
            "FastICA n_components must be positive."))
        max_iterations > 0 || throw(InvalidHyperparameterError(
            "FastICA max_iterations must be positive."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "FastICA tolerance must be finite and positive."))
        new(Int(n_components), Int(max_iterations), Float64(tolerance))
    end
end

struct FittedFastICA{M,T,R,S} <: AbstractFittedTransformer
    model::M
    means::Vector{T}
    unmixing::Matrix{T}
    mixing::Matrix{T}
    sources::Matrix{T}
    iterations::Int
    converged::Bool
    report::R
    schema::S
end

capabilities(::Type{<:FastICA}) = (task=:transformation, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _symmetric_decorrelation(matrix)
    decomposition = svd(matrix)
    decomposition.U * decomposition.Vt
end

function fit(model::FastICA, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "FastICA fitting")
    _validate_numeric_matrix(X, "FastICA")
    n, p = size(X)
    n > 1 && p > 0 || throw(UnsupportedDataError(
        "FastICA requires at least two observations and one feature."))
    model.n_components <= min(n, p) || throw(UnsupportedDataError(
        "FastICA n_components cannot exceed the smaller input dimension."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    means = vec(mean(data; dims=1))
    centered = data .- transpose(means)
    covariance = Symmetric(transpose(centered) * centered / T(n))
    decomposition = eigen(covariance)
    ordering = sortperm(decomposition.values; rev=true)[1:model.n_components]
    values = decomposition.values[ordering]
    minimum(values) > eps(T) || throw(NumericalFailureError(
        "FastICA whitening covariance is rank deficient; reduce n_components."))
    whitening = Diagonal(inv.(sqrt.(values))) *
        transpose(decomposition.vectors[:, ordering])
    whitened = centered * transpose(whitening)
    rng = derive_context(context, :fastica, :initialization).rng
    rotation = _symmetric_decorrelation(
        randn(rng, T, model.n_components, model.n_components))
    convergence_history = T[]
    converged = false
    iterations = effective_max_iterations(context, model.max_iterations)
    tolerance = T(effective_tolerance(context, model.tolerance))
    for iteration in 1:iterations
        activations = whitened * transpose(rotation)
        nonlinear = tanh.(activations)
        derivative_means = vec(mean(one(T) .- abs2.(nonlinear); dims=1))
        updated = transpose(nonlinear) * whitened / T(n) .-
            Diagonal(derivative_means) * rotation
        updated = _symmetric_decorrelation(updated)
        convergence = maximum(abs.(abs.(diag(updated * transpose(rotation))) .- one(T)))
        push!(convergence_history, convergence)
        rotation = updated
        if convergence <= tolerance
            converged = true
            iterations = iteration
            break
        end
    end
    unmixing = rotation * whitening
    sources = centered * transpose(unmixing)
    for component in axes(unmixing, 1)
        pivot = argmax(abs.(view(unmixing, component, :)))
        if unmixing[component, pivot] < 0
            unmixing[component, :] .*= -one(T)
            sources[:, component] .*= -one(T)
        end
    end
    mixing = Matrix{T}(pinv(unmixing))
    details = (components=model.n_components, iterations, converged,
               convergence_history, whitening=:eigen,
               decorrelation=:svd, nonlinearity=:tanh)
    FittedFastICA(model, means, unmixing, mixing, sources, iterations, converged,
        FitReport(status=converged ? :success : :max_iterations,
            observations=n, features=p,
            warnings=converged ? String[] : ["FastICA reached max_iterations."],
            details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedFastICA, X::AbstractMatrix)
    _validate_numeric_matrix(X, "FastICA")
    _validate_feature_count(fitted.schema, X, "FastICA")
    (X .- transpose(fitted.means)) * transpose(fitted.unmixing)
end

function inverse_transform(fitted::FittedFastICA, sources::AbstractMatrix)
    size(sources, 2) == size(fitted.unmixing, 1) || throw(SchemaMismatchError(
        "FastICA source width must match n_components."))
    sources * transpose(fitted.mixing) .+ transpose(fitted.means)
end

report(fitted::FittedFastICA) = fitted.report
