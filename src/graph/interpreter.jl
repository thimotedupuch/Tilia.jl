fit_graph(graph::SemanticGraph, X, y; context, weights=nothing) =
    fit_graph_backend(context.backend, graph, X, y, context, weights)

function fit_graph_backend(::CPUBackend, graph::SemanticGraph, X, y, context, weights=nothing)
    validate_leakage(graph)
    validate_backend(graph, :cpu)
    value = X
    fitted_nodes = AbstractFittedEstimator[]
    node_timings = NamedTuple[]
    fit_execution = lower_graph(graph, X; phase=:fit, device=:cpu)
    inference_execution = lower_graph(graph, X; phase=:inference, device=:cpu)
    for numerical_node in fit_execution.nodes
        node = graph.nodes[numerical_node.semantic_node_id]
        input_value = value
        _record_input!(numerical_node, value)
        node_context = derive_context(context, :graph_node, node.id)
        started = time_ns()
        if numerical_node.operation === :fit_transform
            fitted = fit(node.model, value; context=node_context)
            push!(fitted_nodes, fitted)
            value = transform(fitted, value)
            _record_output!(numerical_node, value)
            _record_primitive_region!(fit_execution, node.id, input_value, value)
        else
            if consumes_target(node)
                y === nothing && throw(UnsupportedDataError(
                    "Predictor node $(node.id) requires a target."))
                fitted = fit(node.model, value, y; weights=weights, context=node_context)
            else
                fitted = fit(node.model, value; context=node_context)
            end
            push!(fitted_nodes, fitted)
            numerical_node.output_shape = ()
            _record_primitive_region!(fit_execution, node.id, input_value, fitted)
        end
        inference_node = inference_execution.nodes[numerical_node.id]
        inference_node.input_shape = numerical_node.input_shape
        inference_node.element_type = numerical_node.element_type
        inference_node.representation = numerical_node.representation
        inference_node.output_shape = node isa TransformNode ? numerical_node.output_shape : (size(X, 1),)
        _record_primitive_region!(inference_execution, node.id, input_value,
                                  node isa TransformNode ? value : y)
        if node isa PredictorNode
            primitive_indices = findall(primitive -> primitive.semantic_node_id == node.id,
                                        inference_execution.primitives)
            isempty(primitive_indices) ||
                (inference_execution.primitives[last(primitive_indices)].output_shape =
                    inference_node.output_shape)
        end
        push!(node_timings, (node_id=node.id, kind=Symbol(nameof(typeof(node.model))),
                             nanoseconds=time_ns() - started))
    end
    FitReport(observations=size(X, 1), features=size(X, 2),
              details=(nodes=length(graph.nodes), execution=:numerical_graph,
                       weighted=weights !== nothing,
                       fit_execution_graph=fit_execution,
                       inference_execution_graph=inference_execution,
                       lowered_primitives=length(fit_execution.primitives),
                       node_timings=node_timings), context=context) |>
        report -> FittedGraph(graph, fitted_nodes, report)
end


function fit_graph_backend(backend::AbstractBackend, graph::SemanticGraph, X, y, context, weights=nothing)
    throw(UnsupportedBackendError(
        "No graph execution extension is loaded for backend $(typeof(backend)); load the corresponding optional package or use CPUBackend."))
end

function _execute_inference_graph(fitted::FittedGraph, X, operation::Symbol)
    execution = lower_graph(fitted.graph, X; phase=:inference,
                            operation=operation, device=:cpu)
    value = X
    for numerical_node in execution.nodes
        node = fitted.fitted_nodes[numerical_node.semantic_node_id]
        input_value = value
        _record_input!(numerical_node, value)
        value = numerical_node.operation === :transform ? transform(node, value) :
                numerical_node.operation === :predict_proba ? predict_proba(node, value) :
                predict(node, value)
        _record_output!(numerical_node, value)
        _record_primitive_region!(execution, numerical_node.semantic_node_id,
                                  input_value, value)
    end
    (output=value, graph=execution)
end

predict_graph(fitted::FittedGraph, X) =
    _execute_inference_graph(fitted, X, :predict).output

predict_proba_graph(fitted::FittedGraph, X) =
    _execute_inference_graph(fitted, X, :predict_proba).output
