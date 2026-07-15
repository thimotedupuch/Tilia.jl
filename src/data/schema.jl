"""Metadata for one feature column."""
struct ColumnSchema
    name::Symbol
    logical_type::Symbol
    physical_type::Type
    allows_missing::Bool
    role::Symbol
    levels::Vector{Any}
    ordered::Bool
    unknown_policy::Symbol
    missing_policy::Symbol
    code_type::Type
    provenance::Vector{Symbol}
    generated_name::Union{Nothing,Symbol}
end

ColumnSchema(name::Symbol, logical_type::Symbol, physical_type::Type,
             allows_missing::Bool, role::Symbol; levels=Any[], ordered=false,
             unknown_policy=:error,
             missing_policy=allows_missing ? :allow : :forbid,
             code_type=Int, provenance=Symbol[name], generated_name=nothing) =
    ColumnSchema(name, logical_type, physical_type, allows_missing, role,
                 Any[levels...], ordered, unknown_policy, missing_policy,
                 code_type, Symbol[provenance...], generated_name)

"""Ordered feature metadata; observations always occupy rows."""
struct Schema
    columns::Vector{ColumnSchema}
    target_name::Union{Nothing,Symbol}
    class_order::Vector{Any}
    target_logical_type::Union{Nothing,Symbol}
    target_physical_type::Union{Nothing,Type}
    target_allows_missing::Bool
end

Schema(columns; target_name=nothing, class_order=Any[], target_logical_type=nothing,
       target_physical_type=nothing, target_allows_missing=false) =
    Schema(collect(columns), target_name, collect(class_order), target_logical_type,
           target_physical_type, target_allows_missing)

# Positional compatibility for version-1 structural persistence payloads.
Schema(columns::Vector{ColumnSchema}, target_name::Union{Nothing,Symbol},
       class_order::Vector{Any}) = Schema(columns; target_name, class_order)

function infer_schema(X::AbstractMatrix)
    T = eltype(X)
    physical = Base.nonmissingtype(T)
    columns = [ColumnSchema(Symbol("x", j), :continuous, physical,
                            Missing <: T, :feature) for j in axes(X, 2)]
    Schema(columns)
end

nfeatures(schema::Schema) = length(schema.columns)

"""Return a schema enriched with supervised-target metadata."""
function with_target(schema::Schema, target::AbstractVector;
                     target_name::Symbol=:target, class_order=Any[])
    T = eltype(target)
    physical = Base.nonmissingtype(T)
    logical = physical <: Number ? :continuous : :categorical
    Schema(schema.columns;
        target_name=schema.target_name === nothing ? target_name : schema.target_name,
        class_order=class_order,
        target_logical_type=logical,
        target_physical_type=physical,
        target_allows_missing=Missing <: T || any(ismissing, target))
end

function with_class_target(schema::Schema, class_order::AbstractVector;
                           target_name::Symbol=:target)
    isempty(class_order) && return schema
    physical = typeof(first(class_order))
    Schema(schema.columns;
        target_name=schema.target_name === nothing ? target_name : schema.target_name,
        class_order=Any[class_order...], target_logical_type=:categorical,
        target_physical_type=physical, target_allows_missing=false)
end
