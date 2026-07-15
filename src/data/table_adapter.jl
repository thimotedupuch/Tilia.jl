function _logical_type(::Type{T}) where {T}
    physical = Base.nonmissingtype(T)
    physical <: Number ? :continuous : :categorical
end

"""Convert any Tables.jl-compatible source into Tilia-owned column storage."""
function column_table(source)
    source isa ColumnTable && return source
    Tables.istable(typeof(source)) || throw(UnsupportedDataError(
        "input of type $(typeof(source)) is neither an AbstractMatrix nor a Tables.jl-compatible table."))
    source_columns = Tables.columntable(source)
    names = Tuple(Symbol.(propertynames(source_columns)))
    raw_columns = map(name -> collect(getproperty(source_columns, name)), names)
    columns = map(raw_columns) do column
        _logical_type(eltype(column)) === :categorical ? categorical_column(column) : column
    end
    schema_columns = ColumnSchema[]
    for (name, raw_column, column) in zip(names, raw_columns, columns)
        T = eltype(raw_column)
        logical_type = column isa CategoricalColumn ? :categorical : :continuous
        allows_missing = Missing <: T || any(ismissing, raw_column)
        if column isa CategoricalColumn
            push!(schema_columns, ColumnSchema(name, logical_type, Base.nonmissingtype(T),
                allows_missing, :feature; levels=column.pool, ordered=column.ordered,
                unknown_policy=column.unknown_policy,
                missing_policy=allows_missing ? :allow : :forbid,
                code_type=eltype(column.codes)))
        else
            push!(schema_columns, ColumnSchema(name, logical_type, Base.nonmissingtype(T),
                allows_missing, :feature;
                missing_policy=allows_missing ? :allow : :forbid))
        end
    end
    ColumnTable(names, columns, Schema(schema_columns))
end

Tables.istable(::Type{<:ColumnTable}) = true
Tables.columnaccess(::Type{<:ColumnTable}) = true
Tables.columns(table::ColumnTable) = table
Tables.columnnames(table::ColumnTable) = table.names
Tables.getcolumn(table::ColumnTable, index::Int) = table.columns[index]
Tables.getcolumn(table::ColumnTable, name::Symbol) = table.columns[findfirst(==(name), table.names)]
