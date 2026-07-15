function _show_type_name(io, value)
    print(io, nameof(typeof(value)))
end

function Base.show(io::IO, model::AbstractEstimator)
    _show_type_name(io, model)
    print(io, '(')
    for index in 1:fieldcount(typeof(model))
        index > 1 && print(io, ", ")
        print(io, fieldname(typeof(model), index), '=')
        show(io, getfield(model, index))
    end
    print(io, ')')
end

function Base.show(io::IO, ::MIME"text/plain", model::AbstractEstimator)
    show(io, model)
    declared = capabilities(model)
    print(io, "\n  task: ", declared.task,
          " | sparse: ", declared.sparse,
          " | missing: ", declared.missing,
          " | weights: ", declared.weights,
          " | partial_fit: ", declared.partial_fit)
end

function Base.show(io::IO, report::FitReport)
    print(io, "FitReport(status=:", report.status,
          ", observations=", report.observations,
          ", features=", report.features,
          ", backend=:", report.backend, ')')
end

function Base.show(io::IO, ::MIME"text/plain", report::FitReport)
    print(io, "FitReport\n",
          "  status: ", report.status, '\n',
          "  data: ", report.observations, " observations × ", report.features,
          " features\n",
          "  backend: ", report.backend,
          " | deterministic: ", report.deterministic,
          " | threads: ", report.thread_count, '\n',
          "  stream: ", report.stream_id, " (seed ", report.root_seed, ")")
    isempty(report.warnings) || print(io, "\n  warnings: ", join(report.warnings, "; "))
    isempty(keys(report.details)) || print(io, "\n  details: ", join(keys(report.details), ", "))
end

function Base.show(io::IO, fitted::AbstractFittedEstimator)
    _show_type_name(io, fitted)
    report_value = report(fitted)
    print(io, "(", report_value.observations, '×', report_value.features,
          ", status=:", report_value.status, ", backend=:",
          report_value.backend, ')')
end

function Base.show(io::IO, ::MIME"text/plain", fitted::AbstractFittedEstimator)
    report_value = report(fitted)
    _show_type_name(io, fitted)
    print(io, '\n', "  model: ")
    hasfield(typeof(fitted), :model) ? show(io, getfield(fitted, :model)) :
        print(io, "semantic graph")
    print(io, '\n', "  data: ", report_value.observations,
          " observations × ", report_value.features, " features\n",
          "  status: ", report_value.status, " | backend: ", report_value.backend)
    isempty(report_value.warnings) ||
        print(io, "\n  warnings: ", join(report_value.warnings, "; "))
end

function Base.show(io::IO, graph::FittedGraph)
    print(io, "FittedGraph(nodes=", length(graph.fitted_nodes),
          ", status=:", graph.report.status,
          ", backend=:", graph.report.backend, ')')
end

function Base.show(io::IO, ::MIME"text/plain", graph::FittedGraph)
    print(io, "FittedGraph with ", length(graph.fitted_nodes), " nodes\n")
    for (index, node) in enumerate(graph.fitted_nodes)
        print(io, "  ", index, ". ", nameof(typeof(node)),
              " [", capabilities(getfield(node, :model)).task, "]\n")
    end
    print(io, "  status: ", graph.report.status,
          " | backend: ", graph.report.backend,
          " | stream: ", graph.report.stream_id)
end

function Base.show(io::IO, column::ColumnSchema)
    print(io, column.name, "::", column.physical_type,
          " [", column.logical_type, ", ", column.role,
          column.allows_missing ? ", missing" : "", ']')
end

function Base.show(io::IO, schema::Schema)
    print(io, "Schema(", nfeatures(schema), " features")
    schema.target_name === nothing || print(io, ", target=:", schema.target_name)
    print(io, ')')
end


function Base.show(io::IO, ::MIME"text/plain", schema::Schema)
    print(io, "Schema with ", nfeatures(schema), " features")
    schema.target_name === nothing || print(io, " and target :", schema.target_name)
    for (index, column) in enumerate(schema.columns)
        print(io, "\n  ", index, ". ")
        show(io, column)
    end
    isempty(schema.class_order) || print(io, "\n  class order: ", schema.class_order)
end

function Base.show(io::IO, graph::NumericalExecutionGraph)
    print(io, "NumericalExecutionGraph(phase=:", graph.phase,
          ", nodes=", length(graph.nodes),
          ", primitives=", length(graph.primitives),
          ", peak_buffers=", graph.peak_buffers, ')')
end

function Base.show(io::IO, ::MIME"text/plain", graph::NumericalExecutionGraph)
    print(io, "NumericalExecutionGraph [", graph.phase, "]\n",
          "  nodes: ", length(graph.nodes), " | primitives: ",
          length(graph.primitives), " | peak buffers: ", graph.peak_buffers)
    for node in graph.nodes
        print(io, "\n  ", node.id, ". ", node.operation,
              " ", node.input_shape, " → ", node.output_shape,
              " [", node.device, "]")
    end
end

function Base.show(io::IO, trace::ExecutionTrace)
    print(io, "ExecutionTrace(nodes=", length(trace.nodes),
          ", total_ms=", round(trace.total_nanoseconds / 1e6; digits=3), ')')
end

function Base.show(io::IO, ::MIME"text/plain", trace::ExecutionTrace)
    print(io, "ExecutionTrace\n  total: ",
          round(trace.total_nanoseconds / 1e6; digits=3), " ms")
    for node in trace.nodes
        print(io, "\n  ", node.node_id, ". ", node.operation, " — ",
              round(node.nanoseconds / 1e6; digits=3), " ms, ",
              node.input_shape, " → ", node.output_shape, ", ",
              Base.format_bytes(node.output_bytes))
    end
end

function Base.show(io::IO, result::ConfusionMatrix)
    print(io, "ConfusionMatrix(", length(result.labels), " classes, ",
          sum(result.matrix), " observations)")
end

function Base.show(io::IO, result::Union{ROCResult,PrecisionRecallResult})
    _show_type_name(io, result)
    print(io, '(', length(result.thresholds), " thresholds)")
end

function Base.show(io::IO, result::CalibrationResult)
    print(io, "CalibrationResult(", length(result.counts), " populated bins)")
end

function Base.show(io::IO, result::PermutationImportanceResult)
    print(io, "PermutationImportanceResult(", length(result.feature_names),
          " features × ", size(result.importances, 2), " repeats)")
end

function Base.show(io::IO, result::CrossValidationResult)
    print(io, "CrossValidationResult(", length(result.scores), " folds, mean=",
          round(mean(result.scores); digits=4), ')')
end

function Base.show(io::IO, result::TuningResult)
    print(io, "TuningResult(trials=", length(result.trials),
          ", best_score=", result.best_score,
          ", best_parameters=", result.best_parameters, ')')
end

function Base.show(io::IO, ::MIME"text/plain",
                   result::Union{ConfusionMatrix,ROCResult,PrecisionRecallResult,
                                 CalibrationResult,PermutationImportanceResult,
                                 CrossValidationResult,TuningResult})
    show(io, result)
end
