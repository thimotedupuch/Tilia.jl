"""
One-hot encode categorical columns in a native `ColumnTable`, optionally
passing numeric columns through. Unknown inference levels either error or map
to an all-zero block.
"""
struct OneHotEncode{T<:AbstractFloat} <: AbstractTransformer
    handle_unknown::Symbol
    passthrough_numeric::Bool
    output_type::Type{T}
    function OneHotEncode(; handle_unknown::Symbol=:error, passthrough_numeric::Bool=true,
                          output_type::Type{T}=Float64) where {T<:AbstractFloat}
        handle_unknown in (:error, :ignore) || throw(InvalidHyperparameterError(
            "OneHotEncode handle_unknown must be :error or :ignore."))
        new{T}(handle_unknown, passthrough_numeric, output_type)
    end
end

struct OneHotColumnSpec
    name::Symbol
    logical_type::Symbol
    levels::Vector{Any}
    output_names::Vector{Symbol}
end

struct FittedOneHotEncode{M,S,R,SC} <: AbstractFittedTransformer
    model::M
    columns::S
    report::R
    schema::SC
end

capabilities(::Type{<:OneHotEncode}) = (
    task=:transformation, sparse=false, missing=false, weights=false,
    partial_fit=false, probabilistic=false,
)

function fit(model::OneHotEncode, table::ColumnTable; context=default_context())
    require_cpu(context, "OneHotEncode fitting")
    specs = OneHotColumnSpec[]
    output_schema_columns = ColumnSchema[]
    for (name, column, metadata) in zip(table.names, table.columns, table.schema.columns)
        any(ismissing, column) && throw(UnsupportedDataError(
            "OneHotEncode column $name contains missing values; fit Impute first."))
        if metadata.logical_type === :categorical
            levels = column isa CategoricalColumn ? Any[column.pool...] : Any[sort!(unique(column))...]
            names = [Symbol(name, "__", string(level)) for level in levels]
            push!(specs, OneHotColumnSpec(name, :categorical, levels, names))
            append!(output_schema_columns,
                [ColumnSchema(output_name, :continuous, model.output_type, false, :feature;
                              provenance=[name], generated_name=output_name)
                 for output_name in names])
        elseif model.passthrough_numeric
            push!(specs, OneHotColumnSpec(name, :continuous, Any[], [name]))
            push!(output_schema_columns,
                  ColumnSchema(name, :continuous, model.output_type, false, :feature;
                               provenance=metadata.provenance,
                               generated_name=metadata.generated_name))
        else
            push!(specs, OneHotColumnSpec(name, :ignored, Any[], Symbol[]))
        end
    end
    isempty(output_schema_columns) && throw(UnsupportedDataError(
        "OneHotEncode produced no features; enable passthrough_numeric or provide categorical columns."))
    output_schema = Schema(output_schema_columns)
    FittedOneHotEncode(model, specs,
        FitReport(observations=nrows(table), features=length(output_schema_columns),
                  details=(input_features=nfeatures(table), output_features=length(output_schema_columns),
                           handle_unknown=model.handle_unknown), context=context), output_schema)
end

function transform(fitted::FittedOneHotEncode, table::ColumnTable)
    table.names == Tuple(spec.name for spec in fitted.columns) || throw(SchemaMismatchError(
        "OneHotEncode input column names or ordering do not match the fitted table."))
    T = fitted.model.output_type
    output = zeros(T, nrows(table), nfeatures(fitted.schema))
    output_column = 1
    for (input_column, spec) in zip(table.columns, fitted.columns)
        if spec.logical_type === :continuous
            for row in eachindex(input_column)
                value = input_column[row]
                ismissing(value) && throw(UnsupportedDataError(
                    "OneHotEncode numeric column $(spec.name) contains missing values; transform with Impute first."))
                output[row, output_column] = value
            end
            output_column += 1
        elseif spec.logical_type === :ignored
            continue
        else
            lookup = Dict(level => index for (index, level) in enumerate(spec.levels))
            for row in eachindex(input_column)
                value = input_column[row]
                if ismissing(value) || !haskey(lookup, value)
                    fitted.model.handle_unknown === :error && throw(SchemaMismatchError(
                        "OneHotEncode encountered unknown level $(repr(value)) in column $(spec.name); use handle_unknown=:ignore to map it to zeros."))
                else
                    output[row, output_column + lookup[value] - 1] = one(T)
                end
            end
            output_column += length(spec.levels)
        end
    end
    output
end

transform(fitted::FittedOneHotEncode, source) = transform(fitted, column_table(source))

output_schema(::OneHotEncode, fitted::FittedOneHotEncode) = fitted.schema
report(fitted::FittedOneHotEncode) = fitted.report
