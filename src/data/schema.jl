"""Metadata for one feature column."""
struct ColumnSchema
    name::Symbol
    logical_type::Symbol
    physical_type::Type
    allows_missing::Bool
    role::Symbol
end

"""Ordered feature metadata; observations always occupy rows."""
struct Schema
    columns::Vector{ColumnSchema}
    target_name::Union{Nothing,Symbol}
    class_order::Vector{Any}
end

Schema(columns; target_name=nothing, class_order=Any[]) =
    Schema(collect(columns), target_name, collect(class_order))

function infer_schema(X::AbstractMatrix)
    T = eltype(X)
    physical = Base.nonmissingtype(T)
    columns = [ColumnSchema(Symbol("x", j), :continuous, physical,
                            Missing <: T, :feature) for j in axes(X, 2)]
    Schema(columns)
end

nfeatures(schema::Schema) = length(schema.columns)
