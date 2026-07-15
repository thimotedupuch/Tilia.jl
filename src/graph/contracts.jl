# Feature-count and backend declarations live after estimator definitions so
# graph metadata remains centralized without introducing backend checks in
# individual fitting implementations.
preserves_feature_count(::Union{Standardize,Impute}) = true

backend_compatibility(::Union{Standardize,LogisticRegression}) = (:cpu, :reactant)

function _schema_with_columns(input::Schema, columns)
    Schema(columns; target_name=input.target_name,
           class_order=input.class_order,
           target_logical_type=input.target_logical_type,
           target_physical_type=input.target_physical_type,
           target_allows_missing=input.target_allows_missing)
end

function _generated_columns(prefix::Symbol, count::Integer, physical::Type;
                            provenance=Symbol[])
    [ColumnSchema(Symbol(prefix, index), :continuous, physical, false, :feature;
                  provenance, generated_name=Symbol(prefix, index))
     for index in 1:count]
end

function _schema_indices(schema::Schema, columns)
    keys = _column_keys(columns)
    indices = Int[]
    for key in keys
        index = key isa Symbol ? findfirst(column -> column.name == key, schema.columns) : Int(key)
        index === nothing && throw(SchemaMismatchError(
            "Select column $(repr(key)) is absent from the semantic input schema."))
        1 <= index <= nfeatures(schema) || throw(SchemaMismatchError(
            "Select column index $index is outside 1:$(nfeatures(schema))."))
        push!(indices, index)
    end
    indices
end

output_schema(::Standardize, schema::Schema) = schema

function output_schema(::Impute, schema::Schema)
    columns = [ColumnSchema(column.name, column.logical_type,
        column.physical_type, false, column.role;
        levels=column.levels, ordered=column.ordered,
        unknown_policy=column.unknown_policy, missing_policy=:imputed,
        code_type=column.code_type, provenance=column.provenance,
        generated_name=column.generated_name) for column in schema.columns]
    _schema_with_columns(schema, columns)
end

function output_schema(model::Select, schema::Schema)
    _schema_with_columns(schema, schema.columns[_schema_indices(schema, model.columns)])
end

function output_schema(model::OneHotEncode, schema::Schema)
    columns = ColumnSchema[]
    for metadata in schema.columns
        if metadata.logical_type === :categorical
            isempty(metadata.levels) && throw(SchemaMismatchError(
                "OneHotEncode requires known levels for column $(metadata.name) during graph schema propagation."))
            for level in metadata.levels
                name = Symbol(metadata.name, "__", string(level))
                push!(columns, ColumnSchema(name, :continuous, model.output_type,
                    false, :feature; provenance=[metadata.name], generated_name=name))
            end
        elseif model.passthrough_numeric
            push!(columns, ColumnSchema(metadata.name, :continuous,
                model.output_type, false, :feature;
                provenance=metadata.provenance,
                generated_name=metadata.generated_name))
        end
    end
    isempty(columns) && throw(SchemaMismatchError(
        "OneHotEncode produces no semantic output columns."))
    _schema_with_columns(schema, columns)
end

function _decomposition_schema(model, schema::Schema, observations::Integer)
    count = model isa PCA ? something(model.n_components,
        min(observations, nfeatures(schema))) : model.n_components
    count <= min(observations, nfeatures(schema)) || throw(SchemaMismatchError(
        "$(nameof(typeof(model))) requests $count components but the graph input supports at most $(min(observations, nfeatures(schema)))."))
    physical = float(promote_type((column.physical_type for column in schema.columns)...))
    _schema_with_columns(schema,
        _generated_columns(:component, count, physical;
                           provenance=[column.name for column in schema.columns]))
end

output_schema(model::Union{PCA,TruncatedSVD}, schema::Schema) =
    _decomposition_schema(model, schema, typemax(Int))

function output_schema(model::NearestNeighbors, schema::Schema)
    physical = float(promote_type((column.physical_type for column in schema.columns)...))
    _schema_with_columns(schema,
        _generated_columns(:neighbor_distance, model.n_neighbors, physical;
                           provenance=[column.name for column in schema.columns]))
end

function output_schema(model::BernoulliRBM, schema::Schema)
    physical = float(promote_type((column.physical_type for column in schema.columns)...))
    _schema_with_columns(schema,
        _generated_columns(:hidden, model.n_components, physical;
                           provenance=[column.name for column in schema.columns]))
end

output_schema(model::Parallel, schema::Schema) =
    tuple((output_schema(step, schema) for step in model.steps)...)

function output_schema(::Concatenate, schemas::Tuple)
    isempty(schemas) && throw(SchemaMismatchError(
        "Concatenate requires at least one semantic input schema."))
    all(schema -> schema isa Schema, schemas) || throw(SchemaMismatchError(
        "Concatenate semantic inputs must all be schemas."))
    _schema_with_columns(first(schemas),
        reduce(vcat, [schema.columns for schema in schemas]))
end

function output_schema(model::ColumnMap, schema::Schema)
    outputs = ColumnSchema[]
    for mapping in model.mappings
        selected = _schema_with_columns(schema,
            schema.columns[_schema_indices(schema, first(mapping))])
        append!(outputs, output_schema(last(mapping), selected).columns)
    end
    _schema_with_columns(schema, outputs)
end

function _prediction_schema(model, input::Schema)
    declared = capabilities(model)
    if declared.task === :classification
        physical = something(input.target_physical_type, Any)
        column = ColumnSchema(:prediction, :categorical, physical, false, :prediction;
            levels=input.class_order, provenance=Symbol[:target], generated_name=:prediction)
    elseif declared.task === :regression
        physical = something(input.target_physical_type, Float64)
        column = ColumnSchema(:prediction, :continuous, physical, false, :prediction;
            provenance=Symbol[:target], generated_name=:prediction)
    elseif declared.task === :clustering
        column = ColumnSchema(:cluster, :categorical, Int, false, :prediction;
            provenance=[column.name for column in input.columns], generated_name=:cluster)
    else
        column = ColumnSchema(:anomaly, :continuous, Int, false, :prediction;
            provenance=[column.name for column in input.columns], generated_name=:anomaly)
    end
    Schema([column]; target_name=input.target_name, class_order=input.class_order,
           target_logical_type=input.target_logical_type,
           target_physical_type=input.target_physical_type,
           target_allows_missing=input.target_allows_missing)
end

output_schema(model::AbstractPredictor, schema::Schema) = _prediction_schema(model, schema)
output_schema(model::Union{KMeans,GaussianMixture,IsolationForest}, schema::Schema) =
    _prediction_schema(model, schema)

"""Propagate semantic schemas through every node in topological order."""
function propagate_schema(graph::SemanticGraph, input::Schema;
                          observations::Integer=typemax(Int))
    validate_graph(graph)
    current = input
    outputs = Any[]
    for node in graph.nodes
        model = node.model
        current = model isa Union{PCA,TruncatedSVD} ?
            _decomposition_schema(model, current, observations) :
            output_schema(model, current)
        push!(outputs, current)
    end
    outputs
end
