function _validate_numeric_matrix(X::AbstractMatrix, model_name)
    eltype(X) <: Number || throw(UnsupportedDataError(
        "$model_name requires a numeric matrix; received element type $(eltype(X))."))
    all(isfinite, X) || throw(UnsupportedDataError(
        "$model_name does not support missing, NaN, or infinite feature values; impute them first."))
    X
end

function _validate_feature_count(schema::Schema, X::AbstractMatrix, model_name)
    size(X, 2) == nfeatures(schema) || throw(SchemaMismatchError(
        "$model_name was fitted with $(nfeatures(schema)) features; received $(size(X, 2))."))
end

function fit(model::AbstractEstimator, dataset::Dataset; context=default_context())
    fitted = dataset.target === nothing ? fit(model, dataset.features; context=context) :
        fit(model, dataset.features, dataset.target;
            weights=dataset.weights, context=context)
    _replace_input_schema(fitted, dataset.schema)
end

function _replace_input_schema(fitted::AbstractFittedEstimator, schema::Schema)
    hasfield(typeof(fitted), :schema) || return fitted
    fields = fieldnames(typeof(fitted))
    values = ntuple(index -> fields[index] === :schema ? schema : getfield(fitted, index),
                    fieldcount(typeof(fitted)))
    typeof(fitted)(values...)
end

function _numeric_table_input(source)
    table = source isa ColumnTable ? source : column_table(source)
    _numeric_matrix(table), table.schema
end

function fit(model::AbstractEstimator, source, y::AbstractVector;
             weights=nothing, context=default_context())
    matrix, schema = _numeric_table_input(source)
    fitted = fit(model, matrix, y; weights, context)
    _replace_input_schema(fitted, with_target(schema, y))
end

function fit(model::AbstractEstimator, source; context=default_context())
    matrix, schema = _numeric_table_input(source)
    _replace_input_schema(fit(model, matrix; context), schema)
end

function predict(fitted::AbstractFittedEstimator, source)
    matrix, _ = _numeric_table_input(source)
    predict(fitted, matrix)
end

function transform(fitted::AbstractFittedEstimator, source)
    matrix, _ = _numeric_table_input(source)
    transform(fitted, matrix)
end

function predict_proba(fitted::AbstractFittedEstimator, source)
    matrix, _ = _numeric_table_input(source)
    predict_proba(fitted, matrix)
end

function fit(chain::Chain, input, y=nothing; context=default_context(), weights=nothing)
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    fit_graph(build_graph(chain), X, y; context=context, weights=weights)
end

# Resolve the table-adapter method intersection for supervised chains.
fit(chain::Chain, input, y::AbstractVector; context=default_context(), weights=nothing) =
    fit_graph(build_graph(chain),
        input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input),
        y; context, weights)

function predict(fitted::FittedGraph, input)
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    predict_graph(fitted, X)
end
function transform(fitted::FittedGraph, input)
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    predict_graph(fitted, X)
end
function predict_proba(fitted::FittedGraph, input)
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    predict_proba_graph(fitted, X)
end
report(fitted::FittedGraph) = fitted.report

fit(model::Union{Impute,OneHotEncode}, source; context=default_context()) =
    fit(model, column_table(source); context=context)

function predict_proba(model, args...)
    throw(UnsupportedDataError(
        "$(nameof(typeof(model))) does not declare probabilistic prediction support; inspect capabilities(model).probabilistic and use predict instead."))
end

function partial_fit(model, args...)
    throw(UnsupportedDataError(
        "$(nameof(typeof(model))) does not declare incremental fitting support; inspect capabilities(model).partial_fit and call fit on the complete training data."))
end
