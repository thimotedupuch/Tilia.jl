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

fit(model::AbstractEstimator, dataset::Dataset; context=default_context()) =
    dataset.target === nothing ? fit(model, dataset.features; context=context) :
    fit(model, dataset.features, dataset.target; weights=dataset.weights, context=context)

function fit(chain::Chain, input, y=nothing; context=default_context(), weights=nothing)
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    fit_graph(build_graph(chain), X, y; context=context, weights=weights)
end

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

function predict_proba(model, args...); throw(MethodError(predict_proba, (model, args...))); end
function partial_fit(model, args...); throw(MethodError(partial_fit, (model, args...))); end
