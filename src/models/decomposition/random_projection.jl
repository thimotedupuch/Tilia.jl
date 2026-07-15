"""Deterministic Gaussian or sparse Johnson–Lindenstrauss projection."""
struct RandomProjection <: AbstractTransformer
    n_components::Int
    distribution::Symbol
    density::Union{Nothing,Float64}
    function RandomProjection(; n_components::Integer=2,
                              distribution::Symbol=:gaussian,
                              density=nothing)
        n_components > 0 || throw(InvalidHyperparameterError(
            "RandomProjection n_components must be positive."))
        distribution in (:gaussian, :sparse) || throw(InvalidHyperparameterError(
            "RandomProjection distribution must be :gaussian or :sparse."))
        density === nothing ||
            (density isa Real && isfinite(density) && 0 < density <= 1) ||
            throw(InvalidHyperparameterError(
                "RandomProjection density must be nothing or lie in (0, 1]."))
        new(Int(n_components), distribution,
            density === nothing ? nothing : Float64(density))
    end
end

struct FittedRandomProjection{M,T,R,S} <: AbstractFittedTransformer
    model::M
    projection::Matrix{T}
    report::R
    schema::S
end

capabilities(::Type{<:RandomProjection}) = (task=:transformation, sparse=true,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function fit(model::RandomProjection, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "RandomProjection fitting")
    _validate_numeric_matrix(X, "RandomProjection")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "RandomProjection requires observations and features."))
    model.n_components <= size(X, 2) || throw(UnsupportedDataError(
        "RandomProjection n_components cannot exceed the input feature count."))
    T = float(eltype(X))
    rng = derive_context(context, :random_projection, :matrix).rng
    density = model.density === nothing ? min(1.0, inv(sqrt(size(X, 2)))) : model.density
    projection = if model.distribution === :gaussian
        randn(rng, T, model.n_components, size(X, 2)) ./ sqrt(T(model.n_components))
    else
        scale = inv(sqrt(T(density * model.n_components)))
        matrix = zeros(T, model.n_components, size(X, 2))
        for index in eachindex(matrix)
            draw = rand(rng)
            matrix[index] = draw < density / 2 ? -scale :
                            draw < density ? scale : zero(T)
        end
        matrix
    end
    details = (components=model.n_components, distribution=model.distribution,
               density=model.distribution === :sparse ? density : 1.0,
               nonzero_projection_entries=count(value -> !iszero(value), projection))
    FittedRandomProjection(model, projection,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedRandomProjection, X::AbstractMatrix)
    _validate_numeric_matrix(X, "RandomProjection")
    _validate_feature_count(fitted.schema, X, "RandomProjection")
    X * transpose(fitted.projection)
end

report(fitted::FittedRandomProjection) = fitted.report
