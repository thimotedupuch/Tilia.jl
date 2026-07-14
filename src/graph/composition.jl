struct FittedSelect{M,I,R,S,OS} <: AbstractFittedTransformer
    model::M
    indices::I
    report::R
    schema::S
    output_schema::OS
end

struct FittedParallel{M,F,R,S} <: AbstractFittedTransformer
    model::M
    fitted_steps::F
    report::R
    schema::S
end

struct FittedConcatenate{M,W,R,S} <: AbstractFittedTransformer
    model::M
    widths::W
    report::R
    schema::S
end

struct FittedColumnMap{M,K,F,R,S} <: AbstractFittedTransformer
    model::M
    keys::K
    fitted_steps::F
    report::R
    schema::S
end

capabilities(::Type{<:Union{Select,Parallel,Concatenate,ColumnMap}}) =
    (task=:transformation, sparse=false, missing=true, weights=false,
     partial_fit=false, probabilistic=false)

function _column_keys(columns)
    raw = columns isa Tuple || columns isa AbstractVector || columns isa AbstractRange ?
          collect(columns) : Any[columns]
    length(raw) == 1 && first(raw) isa Union{Tuple,AbstractVector,AbstractRange} ?
        collect(first(raw)) : raw
end

function _resolve_columns(table::ColumnTable, columns)
    keys = _column_keys(columns)
    indices = Int[]
    for key in keys
        index = key isa Symbol ? findfirst(==(key), table.names) : Int(key)
        index === nothing && throw(SchemaMismatchError("Select column $(repr(key)) is absent."))
        1 <= index <= nfeatures(table) || throw(SchemaMismatchError(
            "Select column index $index is outside 1:$(nfeatures(table))."))
        push!(indices, index)
    end
    indices
end

function _resolve_columns(X::AbstractMatrix, columns)
    keys = _column_keys(columns)
    all(key -> key isa Integer, keys) || throw(SchemaMismatchError(
        "Matrix Select columns must be integer indices."))
    indices = Int.(keys)
    all(index -> 1 <= index <= size(X, 2), indices) || throw(SchemaMismatchError(
        "Select contains a column outside 1:$(size(X, 2))."))
    indices
end

function _select_columns(table::ColumnTable, indices)
    names = Tuple(table.names[index] for index in indices)
    columns = Tuple(table.columns[index] for index in indices)
    schema = Schema(table.schema.columns[indices])
    ColumnTable(names, columns, schema)
end
_select_columns(X::AbstractMatrix, indices) = X[:, indices]

function fit(model::Select, input::Union{AbstractMatrix,ColumnTable}; context=default_context())
    require_cpu(context, "Select fitting")
    indices = _resolve_columns(input, model.columns)
    output = _select_columns(input, indices)
    FittedSelect(model, indices,
        FitReport(observations=size(input, 1), features=length(indices),
                  details=(selected_columns=copy(indices),)),
        infer_schema(input), infer_schema(output))
end

function transform(fitted::FittedSelect, input::Union{AbstractMatrix,ColumnTable})
    _validate_feature_count(fitted.schema, input, "Select")
    _select_columns(input, fitted.indices)
end
report(fitted::FittedSelect) = fitted.report
_validate_feature_count(schema::Schema, table::ColumnTable, model_name) =
    nfeatures(table) == nfeatures(schema) ? nothing : throw(SchemaMismatchError(
        "$model_name was fitted with $(nfeatures(schema)) features; received $(nfeatures(table))."))

function fit(model::Parallel, input; context=default_context())
    require_cpu(context, "Parallel fitting")
    fitted_steps = map(step -> fit(step, input; context=context), model.steps)
    outputs = map(step -> transform(step, input), fitted_steps)
    FittedParallel(model, fitted_steps,
        FitReport(observations=size(input, 1), features=sum(output -> size(output, 2), outputs),
                  details=(branches=length(model.steps), output_widths=map(output -> size(output, 2), outputs))),
        infer_schema(input))
end
transform(fitted::FittedParallel, input) =
    map(step -> transform(step, input), fitted.fitted_steps)
report(fitted::FittedParallel) = fitted.report

function _numeric_matrix(X::AbstractMatrix)
    _validate_numeric_matrix(X, "Concatenate")
    Matrix(X)
end
function _numeric_matrix(table::ColumnTable)
    all(metadata -> metadata.logical_type === :continuous, table.schema.columns) ||
        throw(UnsupportedDataError("Concatenate requires categorical table branches to be encoded first."))
    all(column -> !any(ismissing, column), table.columns) || throw(UnsupportedDataError(
        "Concatenate requires missing table values to be imputed first."))
    reduce(hcat, [collect(column) for column in table.columns])
end

function _concatenate_inputs(inputs::Tuple)
    isempty(inputs) && throw(UnsupportedDataError("Concatenate requires at least one input."))
    matrices = map(_numeric_matrix, inputs)
    rows = size(first(matrices), 1)
    all(matrix -> size(matrix, 1) == rows, matrices) || throw(SchemaMismatchError(
        "Concatenate branches must have equal observation counts."))
    reduce(hcat, matrices)
end

function fit(model::Concatenate, inputs::Tuple; context=default_context())
    require_cpu(context, "Concatenate fitting")
    output = _concatenate_inputs(inputs)
    FittedConcatenate(model, map(input -> size(input, 2), inputs),
        FitReport(observations=size(output, 1), features=size(output, 2),
                  details=(branches=length(inputs), output_features=size(output, 2))),
        infer_schema(output))
end
transform(fitted::FittedConcatenate, inputs::Tuple) = _concatenate_inputs(inputs)
report(fitted::FittedConcatenate) = fitted.report

function _mapping_selection(input, key)
    indices = _resolve_columns(input, key)
    _select_columns(input, indices)
end

_mapping_input(selection, ::Union{Impute,OneHotEncode}) = selection
_mapping_input(selection::ColumnTable, ::Union{Impute,OneHotEncode}) = selection
_mapping_input(selection::ColumnTable, ::AbstractTransformer) = _numeric_matrix(selection)
_mapping_input(selection, ::AbstractTransformer) = selection

function fit(model::ColumnMap, input::Union{AbstractMatrix,ColumnTable}; context=default_context())
    require_cpu(context, "ColumnMap fitting")
    keys = map(first, model.mappings)
    fitted_steps = map(model.mappings) do mapping
        selection = _mapping_selection(input, first(mapping))
        fit(last(mapping), _mapping_input(selection, last(mapping)); context=context)
    end
    outputs = map(keys, fitted_steps) do key, step
        selection = _mapping_selection(input, key)
        transform(step, _mapping_input(selection, step.model))
    end
    matrix = _concatenate_inputs(outputs)
    FittedColumnMap(model, keys, fitted_steps,
        FitReport(observations=size(input, 1), features=size(matrix, 2),
                  details=(mappings=length(keys), output_features=size(matrix, 2))),
        infer_schema(input))
end

function transform(fitted::FittedColumnMap, input::Union{AbstractMatrix,ColumnTable})
    _validate_feature_count(fitted.schema, input, "ColumnMap")
    outputs = map(fitted.keys, fitted.fitted_steps) do key, step
        selection = _mapping_selection(input, key)
        transform(step, _mapping_input(selection, step.model))
    end
    _concatenate_inputs(outputs)
end
report(fitted::FittedColumnMap) = fitted.report
