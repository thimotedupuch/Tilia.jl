const PERSISTENCE_FORMAT_VERSION = 1

_type_name(::Type{T}) where {T} = string(T)
function _numeric_type(name::AbstractString)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    name == "Int32" && return Int32
    name == "Int64" && return Int64
    throw(PersistenceVersionError("Unsupported persisted numeric type $name."))
end

function _encode_value(value)
    value isa Symbol && return Dict("kind" => "Symbol", "value" => string(value))
    value isa AbstractString && return Dict("kind" => "String", "value" => String(value))
    value isa Bool && return Dict("kind" => "Bool", "value" => value)
    value isa Integer && return Dict("kind" => string(typeof(value)), "value" => string(value))
    value isa AbstractFloat && return Dict("kind" => string(typeof(value)), "value" => repr(value))
    value === nothing && return Dict("kind" => "Nothing", "value" => "")
    throw(ArgumentError("Persistence does not support scalar value type $(typeof(value))."))
end

function _decode_value(data)
    kind = data["kind"]
    kind == "Symbol" && return Symbol(data["value"])
    kind == "String" && return data["value"]
    kind == "Bool" && return data["value"]
    kind == "Nothing" && return nothing
    kind in ("Int32", "Int64", "Float32", "Float64") &&
        return parse(_numeric_type(kind), data["value"])
    throw(PersistenceVersionError("Unsupported persisted scalar kind $kind."))
end

function _encode_schema(schema::Schema)
    Dict(
        "columns" => [Dict("name" => string(column.name),
                           "logical_type" => string(column.logical_type),
                           "physical_type" => string(column.physical_type),
                           "allows_missing" => column.allows_missing,
                           "role" => string(column.role)) for column in schema.columns],
        "target_name" => schema.target_name === nothing ? "" : string(schema.target_name),
        "class_order" => [_encode_value(value) for value in schema.class_order],
    )
end

function _physical_type(name)
    name in ("Float32", "Float64", "Int32", "Int64") && return _numeric_type(name)
    name == "String" && return String
    name == "Symbol" && return Symbol
    name == "Bool" && return Bool
    Any
end

function _decode_schema(data)
    columns = [ColumnSchema(Symbol(column["name"]), Symbol(column["logical_type"]),
                            _physical_type(column["physical_type"]), column["allows_missing"],
                            Symbol(column["role"])) for column in data["columns"]]
    target_name = isempty(data["target_name"]) ? nothing : Symbol(data["target_name"])
    Schema(columns; target_name=target_name,
           class_order=Any[_decode_value(value) for value in data["class_order"]])
end

_host_endianness() = Base.ENDIAN_BOM == 0x04030201 ? "little" : "big"

function _write_array(directory, filename, array::AbstractArray)
    isbitstype(eltype(array)) || throw(ArgumentError(
        "Persistent arrays require an isbits element type; received $(eltype(array))."))
    path = joinpath(directory, filename)
    open(path, "w") do io
        write(io, vec(array))
    end
    Dict("file" => filename, "element_type" => string(eltype(array)),
         "dimensions" => collect(size(array)), "endianness" => _host_endianness())
end


function _read_array(directory, metadata)
    expected_endianness = _host_endianness()
    metadata["endianness"] == expected_endianness || throw(PersistenceVersionError(
        "Model array endianness $(metadata["endianness"]) differs from host $expected_endianness."))
    T = _numeric_type(metadata["element_type"])
    dimensions = Tuple(Int.(metadata["dimensions"]))
    count = prod(dimensions)
    values = Vector{T}(undef, count)
    open(joinpath(directory, metadata["file"]), "r") do io
        read!(io, values)
        eof(io) || throw(PersistenceVersionError("Array $(metadata["file"]) has trailing data."))
    end
    reshape(values, dimensions)
end

function _write_toml(path, data)
    open(path, "w") do io
        TOML.print(io, data; sorted=true)
    end
end

function _persistence_scalar_type(name)
    name == "Any" && return Any
    name == "Bool" && return Bool
    name == "String" && return String
    name == "Symbol" && return Symbol
    name == "Float16" && return Float16
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    name == "Int8" && return Int8
    name == "Int16" && return Int16
    name == "Int32" && return Int32
    name == "Int64" && return Int64
    name == "UInt8" && return UInt8
    name == "UInt16" && return UInt16
    name == "UInt32" && return UInt32
    name == "UInt64" && return UInt64
    throw(PersistenceVersionError("Unsupported persisted scalar type $name."))
end

function _encode_structural(value, arrays_directory, counter)
    value === nothing && return Dict("kind" => "nothing")
    value isa Bool && return Dict("kind" => "scalar", "type" => "Bool", "value" => value)
    value isa Integer && return Dict("kind" => "scalar", "type" => string(typeof(value)),
                                     "value" => string(value))
    value isa AbstractFloat && return Dict("kind" => "scalar", "type" => string(typeof(value)),
                                           "value" => repr(value))
    value isa Symbol && return Dict("kind" => "symbol", "value" => string(value))
    value isa AbstractString && return Dict("kind" => "string", "value" => String(value))
    value isa Type && return Dict("kind" => "type", "name" => string(value))
    if value isa AbstractArray && isbitstype(eltype(value)) &&
            eltype(value) <: Number && eltype(value) !== Bool
        counter[] += 1
        metadata = _write_array(arrays_directory, "array_$(counter[]).bin", value)
        return Dict("kind" => "array", "metadata" => metadata)
    end
    if value isa NamedTuple
        return Dict("kind" => "named_tuple", "names" => string.(collect(keys(value))),
                    "values" => [_encode_structural(item, arrays_directory, counter)
                                 for item in Base.values(value)])
    end
    if value isa Pair
        return Dict("kind" => "pair",
                    "first" => _encode_structural(first(value), arrays_directory, counter),
                    "second" => _encode_structural(last(value), arrays_directory, counter))
    end
    if value isa Tuple
        return Dict("kind" => "tuple", "values" =>
                    [_encode_structural(item, arrays_directory, counter) for item in value])
    end
    if value isa AbstractVector
        return Dict("kind" => "vector", "element_type" => string(eltype(value)),
                    "values" => [_encode_structural(item, arrays_directory, counter) for item in value])
    end
    parentmodule(typeof(value)) === (@__MODULE__) || throw(ArgumentError(
        "Structural persistence does not support value type $(typeof(value))."))
    names = fieldnames(typeof(value))
    Dict("kind" => "struct", "type" => string(nameof(typeof(value))),
         "fields" => string.(collect(names)),
         "values" => [_encode_structural(getfield(value, name), arrays_directory, counter)
                      for name in names])
end

function _decode_scalar(data)
    T = _persistence_scalar_type(data["type"])
    T === Bool && return data["value"]
    T <: Integer && return parse(T, data["value"])
    T <: AbstractFloat && return parse(T, data["value"])
    throw(PersistenceVersionError("Unsupported structural scalar type $(data["type"])."))
end

function _typed_vector(values, element_type_name)
    isempty(values) && return try
        _persistence_scalar_type(element_type_name)[]
    catch
        Any[]
    end
    first_type = typeof(first(values))
    all(value -> typeof(value) === first_type, values) ? first_type[values...] : Any[values...]
end

function _structural_type(name)
    symbol = Symbol(name)
    isdefined(@__MODULE__, symbol) || throw(PersistenceVersionError(
        "Persisted object references unknown Tilia type $name."))
    getfield(@__MODULE__, symbol)
end

function _decode_structural(data, directory)
    kind = data["kind"]
    kind == "nothing" && return nothing
    kind == "scalar" && return _decode_scalar(data)
    kind == "symbol" && return Symbol(data["value"])
    kind == "string" && return data["value"]
    kind == "type" && return _persistence_scalar_type(data["name"])
    kind == "array" && return _read_array(directory, data["metadata"])
    if kind == "named_tuple"
        names = Tuple(Symbol.(data["names"]))
        values = Tuple(_decode_structural(value, directory) for value in data["values"])
        return NamedTuple{names}(values)
    end
    kind == "pair" && return _decode_structural(data["first"], directory) =>
                                  _decode_structural(data["second"], directory)
    kind == "tuple" && return Tuple(_decode_structural(value, directory)
                                    for value in data["values"])
    if kind == "vector"
        values = [_decode_structural(value, directory) for value in data["values"]]
        return _typed_vector(values, data["element_type"])
    end
    kind == "struct" || throw(PersistenceVersionError("Unknown structural persistence kind $kind."))
    type_wrapper = _structural_type(data["type"])
    fields = Symbol.(data["fields"])
    values = [_decode_structural(value, directory) for value in data["values"]]
    if type_wrapper <: AbstractEstimator
        type_wrapper === Chain && return Chain(values[1]...)
        type_wrapper === Parallel && return Parallel(values[1]...)
        type_wrapper === ColumnMap && return ColumnMap(values[1]...)
        type_wrapper === Select && return Select(values[1])
        type_wrapper === Concatenate && return Concatenate()
        parameters = (; (field => value for (field, value) in zip(fields, values))...)
        return type_wrapper(; parameters...)
    end
    if type_wrapper === Schema
        named = Dict(zip(fields, values))
        return Schema(named[:columns]; target_name=named[:target_name],
                      class_order=named[:class_order])
    end
    type_wrapper === SemanticGraph && return SemanticGraph(
        AbstractGraphNode[values[1]...], Tuple{Int,Int}[values[2]...])
    type_wrapper === FittedGraph && return FittedGraph(
        values[1], AbstractFittedEstimator[values[2]...], values[3])
    try
        type_wrapper(values...)
    catch error
        throw(PersistenceVersionError(
            "Could not reconstruct persisted $(data["type"]): $(sprint(showerror, error))."))
    end
end

function _save_structural_model(path::AbstractString, fitted::AbstractFittedEstimator)
    ispath(path) && !isdir(path) && throw(ArgumentError("Model path exists and is not a directory: $path"))
    mkpath(path)
    arrays_directory = joinpath(path, "arrays")
    mkpath(arrays_directory)
    object_path = joinpath(path, "object.toml")
    object = _encode_structural(fitted, arrays_directory, Ref(0))
    _write_toml(object_path, Dict("object" => object))
    artifact_files = ["object.toml"]
    append!(artifact_files, [joinpath("arrays", file) for file in readdir(arrays_directory)])
    checksums = Dict(file => bytes2hex(sha256(read(joinpath(path, file)))) for file in artifact_files)
    _write_toml(joinpath(path, "manifest.toml"), Dict(
        "format_version" => PERSISTENCE_FORMAT_VERSION,
        "estimator" => "GenericFittedEstimator", "estimator_schema_version" => 1,
        "package_version" => string(pkgversion(@__MODULE__)),
        "julia_version" => string(VERSION), "checksums" => checksums))
    path
end

"""Save any standalone fitted estimator using Tilia's structural schema codec."""
save_model(path::AbstractString, fitted::AbstractFittedEstimator) =
    _save_structural_model(path, fitted)

"""Save a fitted estimator in Tilia's versioned, schema-based directory format."""
function save_model(path::AbstractString, fitted::FittedMeanRegressor)
    ispath(path) && !isdir(path) && throw(ArgumentError("Model path exists and is not a directory: $path"))
    mkpath(path)
    specification_path = joinpath(path, "specification.toml")
    schema_path = joinpath(path, "schema.toml")
    _write_toml(specification_path, Dict(
        "estimator" => "MeanRegressor",
        "target_mean" => repr(fitted.mean),
        "target_type" => string(typeof(fitted.mean)),
    ))
    _write_toml(schema_path, Dict(
        "feature_count" => nfeatures(fitted.schema),
        "feature_names" => string.([column.name for column in fitted.schema.columns]),
    ))
    checksums = Dict(
        "specification.toml" => bytes2hex(sha256(read(specification_path))),
        "schema.toml" => bytes2hex(sha256(read(schema_path))),
    )
    _write_toml(joinpath(path, "manifest.toml"), Dict(
        "format_version" => PERSISTENCE_FORMAT_VERSION,
        "estimator" => "MeanRegressor",
        "estimator_schema_version" => 1,
        "package_version" => "0.1.0-DEV",
        "julia_version" => string(VERSION),
        "checksums" => checksums,
    ))
    path
end

function _verify_checksum(path, expected)
    actual = bytes2hex(sha256(read(path)))
    actual == expected || throw(PersistenceVersionError(
        "Checksum mismatch for $(basename(path)); the model artifact may be corrupted."))
end

"""Load a fitted estimator saved by `save_model`, validating its version and checksums."""
function load_model(path::AbstractString)
    manifest_path = joinpath(path, "manifest.toml")
    isfile(manifest_path) || throw(PersistenceVersionError("No manifest.toml found at model path $path."))
    manifest = TOML.parsefile(manifest_path)
    version = get(manifest, "format_version", 0)
    version == PERSISTENCE_FORMAT_VERSION || throw(PersistenceVersionError(
        "Unsupported model format version $version; expected $PERSISTENCE_FORMAT_VERSION."))
    manifest["estimator"] == "FittedGraph" && return _load_graph(path, manifest)
    if manifest["estimator"] == "GenericFittedEstimator"
        for (file, checksum) in manifest["checksums"]
            _verify_checksum(joinpath(path, file), checksum)
        end
        data = TOML.parsefile(joinpath(path, "object.toml"))
        return _decode_structural(data["object"], joinpath(path, "arrays"))
    end
    manifest["estimator"] == "MeanRegressor" || throw(PersistenceVersionError(
        "Unsupported persisted estimator $(manifest["estimator"])."))
    for (file, checksum) in manifest["checksums"]
        _verify_checksum(joinpath(path, file), checksum)
    end
    specification = TOML.parsefile(joinpath(path, "specification.toml"))
    schema_data = TOML.parsefile(joinpath(path, "schema.toml"))
    type_name = specification["target_type"]
    T = type_name == "Float32" ? Float32 : type_name == "Float64" ? Float64 :
        throw(PersistenceVersionError("Unsupported persisted target type $type_name."))
    target_mean = parse(T, specification["target_mean"])
    names = Symbol.(schema_data["feature_names"])
    columns = [ColumnSchema(name, :continuous, T, false, :feature) for name in names]
    schema = Schema(columns)
    fit_report = FitReport(features=length(columns), details=(loaded=true, target_mean=target_mean))
    FittedMeanRegressor(MeanRegressor(), target_mean, fit_report, schema)
end

function _node_kind(node)
    node isa FittedImpute && return "Impute"
    node isa FittedOneHotEncode && return "OneHotEncode"
    node isa FittedStandardize && return "Standardize"
    node isa FittedMeanRegressor && return "MeanRegressor"
    node isa FittedLinearRegressor && return node.model isa RidgeRegression ? "RidgeRegression" : "LinearRegression"
    node isa FittedLogisticRegression && return "LogisticRegression"
    throw(ArgumentError("Persistence is not implemented for fitted node $(typeof(node))."))
end

function _encode_node(node, index, arrays_directory)
    kind = _node_kind(node)
    data = Dict{String,Any}("kind" => kind, "schema" => _encode_schema(node.schema))
    if node isa FittedImpute
        data["model"] = Dict("strategy" => string(node.model.strategy),
                             "fill_value" => _encode_value(node.model.fill_value))
        data["fill_values"] = [_encode_value(value) for value in node.fill_values]
    elseif node isa FittedOneHotEncode
        data["model"] = Dict("handle_unknown" => string(node.model.handle_unknown),
                             "passthrough_numeric" => node.model.passthrough_numeric,
                             "output_type" => string(node.model.output_type))
        data["columns"] = [Dict("name" => string(spec.name),
                                "logical_type" => string(spec.logical_type),
                                "levels" => [_encode_value(level) for level in spec.levels],
                                "output_names" => string.(spec.output_names)) for spec in node.columns]
    elseif node isa FittedStandardize
        data["model"] = Dict("center" => node.model.center, "scale" => node.model.scale)
        data["means"] = _write_array(arrays_directory, "node_$(index)_means.bin", node.means)
        data["scales"] = _write_array(arrays_directory, "node_$(index)_scales.bin", node.scales)
    elseif node isa FittedMeanRegressor
        data["mean"] = _encode_value(node.mean)
    elseif node isa FittedLinearRegressor
        data["model"] = node.model isa RidgeRegression ?
            Dict("lambda" => _encode_value(node.model.lambda),
                 "fit_intercept" => node.model.fit_intercept, "solver" => string(node.model.solver)) :
            Dict("fit_intercept" => node.model.fit_intercept, "solver" => string(node.model.solver))
        data["coefficients"] = _write_array(arrays_directory, "node_$(index)_coefficients.bin", node.coefficients)
        data["intercept"] = _encode_value(node.intercept)
    elseif node isa FittedLogisticRegression
        data["model"] = Dict("lambda" => _encode_value(node.model.lambda),
                             "fit_intercept" => node.model.fit_intercept,
                             "max_iterations" => node.model.max_iterations,
                             "tolerance" => _encode_value(node.model.tolerance))
        data["coefficients"] = _write_array(arrays_directory, "node_$(index)_coefficients.bin", node.coefficients)
        data["intercept"] = _write_array(arrays_directory, "node_$(index)_intercept.bin", node.intercept)
        data["classes"] = [_encode_value(class) for class in node.classes]
    end
    data
end

function _basic_report(node; loaded=false)
    original = report(node)
    FitReport(status=original.status, observations=original.observations,
              features=original.features, backend=original.backend,
              warnings=copy(original.warnings), details=(loaded=loaded,))
end

"""Save a fitted semantic graph and all learned arrays without Julia Serialization."""
function save_model(path::AbstractString, fitted::FittedGraph)
    any(node -> node isa Union{FittedSelect,FittedParallel,FittedConcatenate,
                               FittedColumnMap,FittedDecomposition,FittedKMeans,
                               FittedGaussianNaiveBayes,FittedDiscriminantAnalysis,
                               FittedGaussianMixture,FittedNearestNeighbors,
                               FittedSparseLinearRegressor,FittedSparseLogisticRegression,
                               FittedDecisionTree,FittedForest,FittedHistGradientBoosting,
                               FittedKernelRidge,FittedSupportVectorClassifier,
                               FittedSupportVectorRegressor,FittedMLP,FittedBernoulliRBM},
        fitted.fitted_nodes) && return _save_structural_model(path, fitted)
    ispath(path) && !isdir(path) && throw(ArgumentError("Model path exists and is not a directory: $path"))
    mkpath(path)
    arrays_directory = joinpath(path, "arrays")
    mkpath(arrays_directory)
    nodes = [_encode_node(node, index, arrays_directory)
             for (index, node) in enumerate(fitted.fitted_nodes)]
    specification_path = joinpath(path, "specification.toml")
    schema_path = joinpath(path, "schema.toml")
    report_path = joinpath(path, "report.toml")
    _write_toml(specification_path, Dict("nodes" => nodes))
    input_schema = first(fitted.fitted_nodes).schema
    _write_toml(schema_path, _encode_schema(input_schema))
    _write_toml(report_path, Dict("status" => string(fitted.report.status),
        "observations" => fitted.report.observations, "features" => fitted.report.features,
        "backend" => string(fitted.report.backend), "warnings" => fitted.report.warnings))
    artifact_files = ["specification.toml", "schema.toml", "report.toml"]
    append!(artifact_files, [joinpath("arrays", file) for file in readdir(arrays_directory)])
    checksums = Dict(file => bytes2hex(sha256(read(joinpath(path, file)))) for file in artifact_files)
    _write_toml(joinpath(path, "manifest.toml"), Dict(
        "format_version" => PERSISTENCE_FORMAT_VERSION,
        "estimator" => "FittedGraph", "estimator_schema_version" => 1,
        "package_version" => string(pkgversion(@__MODULE__)), "julia_version" => string(VERSION),
        "graph_structure" => [_node_kind(node) for node in fitted.fitted_nodes],
        "checksums" => checksums))
    path
end

function _decode_node(data, arrays_directory)
    kind = data["kind"]
    schema = _decode_schema(data["schema"])
    if kind == "Impute"
        model_data = data["model"]
        model = Impute(strategy=Symbol(model_data["strategy"]),
                       fill_value=_decode_value(model_data["fill_value"]))
        fills = Tuple(_decode_value(value) for value in data["fill_values"])
        placeholder = FittedImpute(model, fills, FitReport(features=nfeatures(schema)), schema)
    elseif kind == "OneHotEncode"
        model_data = data["model"]
        model = OneHotEncode(handle_unknown=Symbol(model_data["handle_unknown"]),
            passthrough_numeric=model_data["passthrough_numeric"],
            output_type=_numeric_type(model_data["output_type"]))
        specs = [OneHotColumnSpec(Symbol(spec["name"]), Symbol(spec["logical_type"]),
                    Any[_decode_value(level) for level in spec["levels"]], Symbol.(spec["output_names"]))
                 for spec in data["columns"]]
        placeholder = FittedOneHotEncode(model, specs, FitReport(features=nfeatures(schema)), schema)
    elseif kind == "Standardize"
        model = Standardize(center=data["model"]["center"], scale=data["model"]["scale"])
        means = vec(_read_array(arrays_directory, data["means"]))
        scales = vec(_read_array(arrays_directory, data["scales"]))
        placeholder = FittedStandardize(model, means, scales, FitReport(features=nfeatures(schema)), schema)
    elseif kind == "MeanRegressor"
        placeholder = FittedMeanRegressor(MeanRegressor(), _decode_value(data["mean"]),
                                           FitReport(features=nfeatures(schema)), schema)
    elseif kind in ("LinearRegression", "RidgeRegression")
        model_data = data["model"]
        model = kind == "LinearRegression" ?
            LinearRegression(fit_intercept=model_data["fit_intercept"], solver=Symbol(model_data["solver"])) :
            RidgeRegression(lambda=_decode_value(model_data["lambda"]),
                fit_intercept=model_data["fit_intercept"], solver=Symbol(model_data["solver"]))
        coefficients = vec(_read_array(arrays_directory, data["coefficients"]))
        placeholder = FittedLinearRegressor(model, coefficients, _decode_value(data["intercept"]),
                                             FitReport(features=nfeatures(schema)), schema)
    elseif kind == "LogisticRegression"
        model_data = data["model"]
        model = LogisticRegression(lambda=_decode_value(model_data["lambda"]),
            fit_intercept=model_data["fit_intercept"], max_iterations=model_data["max_iterations"],
            tolerance=_decode_value(model_data["tolerance"]))
        coefficients = Matrix(_read_array(arrays_directory, data["coefficients"]))
        intercept = vec(_read_array(arrays_directory, data["intercept"]))
        classes = [_decode_value(class) for class in data["classes"]]
        placeholder = FittedLogisticRegression(model, coefficients, intercept, classes,
                                                FitReport(features=nfeatures(schema)), schema)
    else
        throw(PersistenceVersionError("Unsupported persisted graph node $kind."))
    end
    placeholder
end

function _load_graph(path, manifest)
    for (file, checksum) in manifest["checksums"]
        _verify_checksum(joinpath(path, file), checksum)
    end
    specification = TOML.parsefile(joinpath(path, "specification.toml"))
    arrays_directory = joinpath(path, "arrays")
    fitted_nodes = AbstractFittedEstimator[_decode_node(node, arrays_directory)
                                            for node in specification["nodes"]]
    models = tuple((node.model for node in fitted_nodes)...)
    graph = validate_graph(build_graph(Chain(models...)))
    report_data = TOML.parsefile(joinpath(path, "report.toml"))
    fit_report = FitReport(status=Symbol(report_data["status"]),
        observations=report_data["observations"], features=report_data["features"],
        backend=Symbol(report_data["backend"]), warnings=String.(report_data["warnings"]),
        details=(loaded=true, nodes=length(fitted_nodes), execution=:semantic_graph))
    FittedGraph(graph, fitted_nodes, fit_report)
end
