"""Native supervised or unsupervised dataset container."""
struct Dataset{X,Y,W,S}
    features::X
    target::Y
    weights::W
    schema::S
end

function Dataset(features::AbstractMatrix; target=nothing, weights=nothing,
                 schema=infer_schema(features))
    n = size(features, 1)
    target === nothing || length(target) == n || throw(SchemaMismatchError(
        "Dataset target has length $(length(target)); expected $n observations."))
    weights === nothing || length(weights) == n || throw(SchemaMismatchError(
        "Dataset weights have length $(length(weights)); expected $n observations."))
    nfeatures(schema) == size(features, 2) || throw(SchemaMismatchError(
        "Dataset schema has $(nfeatures(schema)) columns; features have $(size(features, 2))."))
    chosen_schema = target === nothing || schema.target_logical_type !== nothing ? schema :
                    with_target(schema, target)
    Dataset(features, target, weights, chosen_schema)
end

function Dataset(source; target=nothing, weights=nothing, schema=nothing)
    features = column_table(source)
    chosen_schema = schema === nothing ? features.schema : schema
    n = nrows(features)
    target === nothing || length(target) == n || throw(SchemaMismatchError(
        "Dataset target has length $(length(target)); expected $n observations."))
    weights === nothing || length(weights) == n || throw(SchemaMismatchError(
        "Dataset weights have length $(length(weights)); expected $n observations."))
    enriched_schema = target === nothing || chosen_schema.target_logical_type !== nothing ?
                      chosen_schema : with_target(chosen_schema, target)
    Dataset(features, target, weights, enriched_schema)
end
