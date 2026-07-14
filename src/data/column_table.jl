"""Tilia's immutable-name, column-oriented native table representation."""
struct ColumnTable{N<:Tuple,C<:Tuple,S<:Schema}
    names::N
    columns::C
    schema::S
    function ColumnTable(names::N, columns::C, schema::S) where {N<:Tuple,C<:Tuple,S<:Schema}
        length(names) == length(columns) == nfeatures(schema) || throw(SchemaMismatchError(
            "ColumnTable names, columns, and schema must have equal feature counts."))
        lengths = length.(columns)
        isempty(lengths) || all(==(first(lengths)), lengths) || throw(SchemaMismatchError(
            "ColumnTable columns must have equal observation counts."))
        new{N,C,S}(names, columns, schema)
    end
end

Base.size(table::ColumnTable) = (nrows(table), length(table.columns))
Base.size(table::ColumnTable, dimension::Integer) = size(table)[dimension]
nrows(table::ColumnTable) = isempty(table.columns) ? 0 : length(first(table.columns))
nfeatures(table::ColumnTable) = length(table.columns)
infer_schema(table::ColumnTable) = table.schema

function select_rows(table::ColumnTable, indices::AbstractVector{<:Integer})
    columns = map(column -> column[indices], table.columns)
    ColumnTable(table.names, columns, table.schema)
end

select_rows(X::AbstractMatrix, indices::AbstractVector{<:Integer}) = X[indices, :]
