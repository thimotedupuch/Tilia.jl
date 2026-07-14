"""
Replace missing values using `:auto`, `:mean`, `:median`, `:most_frequent`, or
`:constant`. `:auto` uses the mean for numeric columns and the most frequent
level for categorical columns.
"""
struct Impute <: AbstractTransformer
    strategy::Symbol
    fill_value
    function Impute(; strategy::Symbol=:auto, fill_value=nothing)
        strategy in (:auto, :mean, :median, :most_frequent, :constant) ||
            throw(InvalidHyperparameterError(
                "Impute strategy must be :auto, :mean, :median, :most_frequent, or :constant."))
        strategy === :constant && fill_value === nothing && throw(InvalidHyperparameterError(
            "Impute strategy=:constant requires fill_value."))
        new(strategy, fill_value)
    end
end

struct FittedImpute{M,F,R,S} <: AbstractFittedTransformer
    model::M
    fill_values::F
    report::R
    schema::S
end

capabilities(::Type{<:Impute}) = (
    task=:transformation, sparse=false, missing=true, weights=false,
    partial_fit=false, probabilistic=false,
)

function _most_frequent(values, name)
    observed = collect(skipmissing(values))
    isempty(observed) && throw(UnsupportedDataError(
        "Impute cannot infer a value for entirely missing column $name; use strategy=:constant."))
    counts = Dict{eltype(observed),Int}()
    order = eltype(observed)[]
    for value in observed
        haskey(counts, value) || push!(order, value)
        counts[value] = get(counts, value, 0) + 1
    end
    order[argmax([counts[value] for value in order])]
end

function _imputation_value(values, logical_type, model, name)
    model.strategy === :constant && return model.fill_value
    strategy = model.strategy === :auto ?
        (logical_type === :continuous ? :mean : :most_frequent) : model.strategy
    observed = collect(skipmissing(values))
    isempty(observed) && throw(UnsupportedDataError(
        "Impute cannot infer a value for entirely missing column $name; use strategy=:constant."))
    if strategy === :mean
        logical_type === :continuous || throw(UnsupportedDataError(
            "Impute strategy=:mean is invalid for categorical column $name."))
        return mean(observed)
    elseif strategy === :median
        logical_type === :continuous || throw(UnsupportedDataError(
            "Impute strategy=:median is invalid for categorical column $name."))
        return median(observed)
    elseif strategy === :most_frequent
        return _most_frequent(values, name)
    end
    throw(InvalidHyperparameterError("unsupported imputation strategy $strategy."))
end

function fit(model::Impute, X::AbstractMatrix; context=default_context())
    require_cpu(context, "Impute fitting")
    Base.nonmissingtype(eltype(X)) <: Number || throw(UnsupportedDataError(
        "matrix imputation requires numeric elements."))
    size(X, 1) > 0 || throw(UnsupportedDataError("Impute requires at least one observation."))
    schema = infer_schema(X)
    fills = tuple((_imputation_value(view(X, :, column), :continuous, model,
                                     schema.columns[column].name) for column in axes(X, 2))...)
    missing_counts = [count(ismissing, view(X, :, column)) for column in axes(X, 2)]
    FittedImpute(model, fills,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=(strategy=model.strategy, missing_counts=missing_counts)), schema)
end

function fit(model::Impute, table::ColumnTable; context=default_context())
    require_cpu(context, "Impute fitting")
    nrows(table) > 0 || throw(UnsupportedDataError("Impute requires at least one observation."))
    fills = tuple((_imputation_value(column, metadata.logical_type, model, metadata.name)
                   for (column, metadata) in zip(table.columns, table.schema.columns))...)
    missing_counts = [count(ismissing, column) for column in table.columns]
    output_columns = [ColumnSchema(metadata.name, metadata.logical_type, metadata.physical_type,
                                   false, metadata.role) for metadata in table.schema.columns]
    output_schema = Schema(output_columns; target_name=table.schema.target_name,
                           class_order=table.schema.class_order)
    FittedImpute(model, fills,
        FitReport(observations=nrows(table), features=nfeatures(table),
                  details=(strategy=model.strategy, missing_counts=missing_counts)), output_schema)
end

function _filled_vector(column, fill_value)
    T = promote_type(Base.nonmissingtype(eltype(column)), typeof(fill_value))
    result = Vector{T}(undef, length(column))
    for index in eachindex(column)
        result[index] = ismissing(column[index]) ? fill_value : column[index]
    end
    result
end

function transform(fitted::FittedImpute, X::AbstractMatrix)
    size(X, 2) == length(fitted.fill_values) || throw(SchemaMismatchError(
        "Impute was fitted with $(length(fitted.fill_values)) columns; received $(size(X, 2))."))
    columns = [_filled_vector(view(X, :, column), fitted.fill_values[column]) for column in axes(X, 2)]
    hcat(columns...)
end

function transform(fitted::FittedImpute, table::ColumnTable)
    table.names == Tuple(column.name for column in fitted.schema.columns) || throw(SchemaMismatchError(
        "Impute input column names or ordering do not match the fitted schema."))
    columns = map(_filled_vector, table.columns, fitted.fill_values)
    ColumnTable(table.names, columns, fitted.schema)
end

transform(fitted::FittedImpute, source) = transform(fitted, column_table(source))

report(fitted::FittedImpute) = fitted.report
