"""Nonnegative matrix factorization using deterministic multiplicative updates."""
struct NMF <: AbstractTransformer
    n_components::Int
    max_iterations::Int
    tolerance::Float64
    function NMF(; n_components::Integer=2, max_iterations::Integer=200,
                 tolerance::Real=1e-4)
        n_components > 0 || throw(InvalidHyperparameterError("NMF n_components must be positive."))
        max_iterations > 0 || throw(InvalidHyperparameterError("NMF max_iterations must be positive."))
        isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
            "NMF tolerance must be finite and nonnegative."))
        new(Int(n_components), Int(max_iterations), Float64(tolerance))
    end
end

struct FittedNMF{M,T,R,S} <: AbstractFittedTransformer
    model::M
    components::Matrix{T}
    embeddings::Matrix{T}
    reconstruction_error::T
    iterations::Int
    converged::Bool
    report::R
    schema::S
end

capabilities(::Type{<:NMF}) = (task=:transformation, sparse=true,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _validate_nmf_data(X, name)
    _validate_numeric_matrix(X, name)
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "$name requires observations and features."))
    all(>=(zero(eltype(X))), X) || throw(UnsupportedDataError(
        "$name requires nonnegative features."))
end

function _nmf_embedding(X, components, iterations)
    T = eltype(components)
    embeddings = fill(sqrt(max(T(mean(X)), eps(T)) / T(size(components, 1))),
                      size(X, 1), size(components, 1))
    gram = components * transpose(components)
    numerator = X * transpose(components)
    for _ in 1:iterations
        embeddings .*= numerator ./ max.(embeddings * gram, eps(T))
    end
    embeddings
end

function fit(model::NMF, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "NMF fitting")
    _validate_nmf_data(X, "NMF")
    model.n_components <= min(size(X)...) || throw(UnsupportedDataError(
        "NMF n_components cannot exceed the smaller input dimension."))
    T = float(eltype(X))
    data = T.(X)
    rng = derive_context(context, :nmf, :initialization).rng
    scale = sqrt(max(T(mean(data)), eps(T)) / T(model.n_components))
    embeddings = max.(rand(rng, T, size(X, 1), model.n_components) .* scale, eps(T))
    components = max.(rand(rng, T, model.n_components, size(X, 2)) .* scale, eps(T))
    history = T[]
    converged = false
    iterations = effective_max_iterations(context, model.max_iterations)
    tolerance = T(effective_tolerance(context, model.tolerance))
    for iteration in 1:iterations
        components .*= (transpose(embeddings) * data) ./ max.(
            transpose(embeddings) * embeddings * components, eps(T))
        embeddings .*= (data * transpose(components)) ./ max.(
            embeddings * (components * transpose(components)), eps(T))
        error = T(norm(data - embeddings * components))
        push!(history, error)
        if length(history) > 1 &&
           abs(history[end - 1] - error) <= tolerance * max(history[end - 1], one(T))
            converged = true
            iterations = iteration
            break
        end
    end
    reconstruction_error = last(history)
    details = (components=model.n_components, iterations, converged,
               reconstruction_error, objective_history=history,
               solver=:multiplicative_updates)
    FittedNMF(model, components, embeddings, reconstruction_error,
        iterations, converged, FitReport(status=converged ? :success : :max_iterations,
            observations=size(X, 1), features=size(X, 2),
            warnings=converged ? String[] : ["NMF reached max_iterations."],
            details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedNMF, X::AbstractMatrix)
    _validate_nmf_data(X, "NMF")
    _validate_feature_count(fitted.schema, X, "NMF")
    T = eltype(fitted.components)
    _nmf_embedding(T.(X), fitted.components, fitted.model.max_iterations)
end

function inverse_transform(fitted::FittedNMF, embeddings::AbstractMatrix)
    size(embeddings, 2) == size(fitted.components, 1) || throw(SchemaMismatchError(
        "NMF embedding width must match n_components."))
    embeddings * fitted.components
end

report(fitted::FittedNMF) = fitted.report
