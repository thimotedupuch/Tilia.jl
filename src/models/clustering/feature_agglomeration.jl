"""Hierarchically merge similar feature profiles and average each final group."""
struct FeatureAgglomeration <: AbstractTransformer
    n_clusters::Int
    linkage::Symbol
    function FeatureAgglomeration(; n_clusters::Integer=2,
                                  linkage::Symbol=:average)
        n_clusters > 0 || throw(InvalidHyperparameterError(
            "FeatureAgglomeration n_clusters must be positive."))
        linkage in (:single, :complete, :average) || throw(InvalidHyperparameterError(
            "FeatureAgglomeration linkage must be :single, :complete, or :average."))
        new(Int(n_clusters), linkage)
    end
end

struct FittedFeatureAgglomeration{M,G,R,S} <: AbstractFittedTransformer
    model::M
    groups::G
    labels::Vector{Int}
    children::Matrix{Int}
    merge_distances::Vector
    report::R
    schema::S
end

capabilities(::Type{<:FeatureAgglomeration}) = (task=:transformation,
    sparse=false, missing=false, weights=false, partial_fit=false,
    probabilistic=false)

function fit(model::FeatureAgglomeration, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "FeatureAgglomeration fitting")
    _validate_numeric_matrix(X, "FeatureAgglomeration")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "FeatureAgglomeration requires observations and features."))
    model.n_clusters <= size(X, 2) || throw(UnsupportedDataError(
        "FeatureAgglomeration n_clusters cannot exceed the feature count."))
    feature_profiles = Matrix(transpose(float.(X)))
    hierarchy = fit(AgglomerativeClustering(
        n_clusters=model.n_clusters, linkage=model.linkage), feature_profiles;
        context=derive_context(context, :feature_agglomeration, :hierarchy))
    labels = hierarchy.labels
    groups = [findall(==(cluster), labels) for cluster in 1:model.n_clusters]
    details = (clusters=model.n_clusters, linkage=model.linkage,
               groups=copy(groups), original_features=size(X, 2))
    FittedFeatureAgglomeration(model, groups, labels, hierarchy.children,
        hierarchy.merge_distances,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

function transform(fitted::FittedFeatureAgglomeration, X::AbstractMatrix)
    _validate_numeric_matrix(X, "FeatureAgglomeration")
    _validate_feature_count(fitted.schema, X, "FeatureAgglomeration")
    T = float(eltype(X))
    output = Matrix{T}(undef, size(X, 1), length(fitted.groups))
    for (cluster, features) in enumerate(fitted.groups)
        output[:, cluster] .= vec(mean(view(X, :, features); dims=2))
    end
    output
end

function inverse_transform(fitted::FittedFeatureAgglomeration,
                           reduced::AbstractMatrix)
    size(reduced, 2) == length(fitted.groups) || throw(SchemaMismatchError(
        "FeatureAgglomeration reduced width must match n_clusters."))
    output = Matrix{eltype(reduced)}(undef, size(reduced, 1),
                                    nfeatures(fitted.schema))
    for (cluster, features) in enumerate(fitted.groups)
        output[:, features] .= reshape(view(reduced, :, cluster), :, 1)
    end
    output
end

report(fitted::FittedFeatureAgglomeration) = fitted.report
