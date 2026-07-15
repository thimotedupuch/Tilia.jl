abstract type AbstractDecompositionModel <: AbstractTransformer end

"""Principal component analysis using a deterministic thin SVD."""
struct PCA <: AbstractDecompositionModel
    n_components::Union{Nothing,Int}
    whiten::Bool
    function PCA(; n_components=nothing, whiten::Bool=false)
        n_components === nothing || n_components > 0 || throw(InvalidHyperparameterError(
            "PCA n_components must be positive or nothing; received $n_components."))
        new(n_components, whiten)
    end
end

"""Low-rank SVD without centering, suitable for sparse-style semantics."""
struct TruncatedSVD <: AbstractDecompositionModel
    n_components::Int
    function TruncatedSVD(; n_components::Integer=2)
        n_components > 0 || throw(InvalidHyperparameterError(
            "TruncatedSVD n_components must be positive; received $n_components."))
        new(Int(n_components))
    end
end

struct FittedDecomposition{M,T,R,S} <: AbstractFittedTransformer
    model::M
    components::Matrix{T}
    mean::Vector{T}
    singular_values::Vector{T}
    explained_variance::Vector{T}
    explained_variance_ratio::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:PCA}) = (task=:transformation, sparse=false, missing=false,
    weights=false, partial_fit=false, probabilistic=false)
capabilities(::Type{<:TruncatedSVD}) = (task=:transformation, sparse=true, missing=false,
    weights=false, partial_fit=false, probabilistic=false)

function _svd_flip!(components)
    for column in axes(components, 2)
        vector = view(components, :, column)
        pivot = argmax(abs.(vector))
        vector[pivot] < 0 && (vector .*= -1)
    end
    components
end

function fit(model::AbstractDecompositionModel, X::AbstractMatrix; context=default_context())
    require_cpu(context, "$(nameof(typeof(model))) fitting")
    _validate_numeric_matrix(X, string(nameof(typeof(model))))
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError(
        "$(nameof(typeof(model))) requires at least one observation and one feature."))
    maximum_components = min(n, p)
    requested = model isa PCA ? something(model.n_components, maximum_components) : model.n_components
    requested <= maximum_components || throw(InvalidHyperparameterError(
        "$(nameof(typeof(model))) requested $requested components, but at most $maximum_components are available."))
    T = float(eltype(X))
    center = model isa PCA
    feature_mean = center ? T.(vec(mean(X; dims=1))) : zeros(T, p)
    prepared = Matrix{T}(X)
    center && (prepared .-= transpose(feature_mean))
    decomposition = svd(prepared; full=false)
    components = _svd_flip!(Matrix{T}(decomposition.V[:, 1:requested]))
    singular_values = T.(decomposition.S[1:requested])
    denominator = max(n - 1, 1)
    explained = singular_values .^ 2 ./ T(denominator)
    total_variance = sum(abs2, prepared) / T(denominator)
    ratio = iszero(total_variance) ? zeros(T, requested) : explained ./ total_variance
    details = (n_components=requested, centered=center,
               whiten=model isa PCA && model.whiten,
               explained_variance_ratio=sum(ratio))
    fit_report = FitReport(observations=n, features=p, backend=:cpu,
                           details=details, context=context)
    FittedDecomposition(model, components, feature_mean, singular_values,
        explained, ratio, fit_report, infer_schema(X))
end

function transform(fitted::FittedDecomposition, X::AbstractMatrix)
    _validate_numeric_matrix(X, string(nameof(typeof(fitted.model))))
    _validate_feature_count(fitted.schema, X, string(nameof(typeof(fitted.model))))
    projected = (X .- transpose(fitted.mean)) * fitted.components
    if fitted.model isa PCA && fitted.model.whiten
        scales = sqrt.(fitted.explained_variance)
        projected ./= transpose(map(scale -> iszero(scale) ? one(scale) : scale, scales))
    end
    projected
end

function inverse_transform(fitted::FittedDecomposition, X::AbstractMatrix)
    size(X, 2) == size(fitted.components, 2) || throw(SchemaMismatchError(
        "$(nameof(typeof(fitted.model))) inverse input has $(size(X, 2)) components; expected $(size(fitted.components, 2))."))
    restored = Matrix(X)
    if fitted.model isa PCA && fitted.model.whiten
        restored .*= transpose(sqrt.(fitted.explained_variance))
    end
    restored * transpose(fitted.components) .+ transpose(fitted.mean)
end

report(fitted::FittedDecomposition) = fitted.report
