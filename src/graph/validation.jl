function validate_graph(graph::SemanticGraph)
    isempty(graph.nodes) && throw(GraphValidationError("A semantic graph cannot be empty."))
    ids = [node.id for node in graph.nodes]
    ids == collect(1:length(ids)) || throw(GraphValidationError("Graph node identifiers must be topologically ordered."))
    for (from, to) in graph.edges
        from < to || throw(GraphValidationError("Graph edge $from → $to is not topologically ordered."))
        1 <= from <= length(ids) && 1 <= to <= length(ids) ||
            throw(GraphValidationError("Graph edge $from → $to refers to an unknown node."))
    end
    predictors = findall(node -> node isa PredictorNode, graph.nodes)
    length(predictors) <= 1 || throw(GraphValidationError("Chain can contain at most one predictor."))
    isempty(predictors) || only(predictors) == length(graph.nodes) ||
        throw(GraphValidationError("A predictor must be the final Chain step."))
    for node in graph.nodes
        contract = node_contract(node)
        isempty(contract.backend_compatibility) && throw(GraphValidationError(
            "Graph node $(node.id) declares no compatible execution backend."))
        node isa Union{TransformNode,PredictorNode} &&
            !contract.input.rows_are_observations && throw(GraphValidationError(
                "Graph node $(node.id) violates Tilia's rows-as-observations contract."))
    end
    graph
end

"""Predecessor node identifiers, in deterministic semantic order."""
function graph_predecessors(graph::SemanticGraph)
    predecessors = [Int[] for _ in graph.nodes]
    for (from, to) in graph.edges
        push!(predecessors[to], from)
    end
    foreach(sort!, predecessors)
    predecessors
end

"""Sink node identifiers, in deterministic semantic order."""
function graph_sinks(graph::SemanticGraph)
    has_consumer = falses(length(graph.nodes))
    for (from, _) in graph.edges
        has_consumer[from] = true
    end
    findall(!, has_consumer)
end

_is_linear_graph(graph::SemanticGraph) =
    graph.edges == [(id, id + 1) for id in 1:length(graph.nodes)-1]

_graph_input(values, predecessors::Vector{Int}, input) =
    isempty(predecessors) ? input :
    length(predecessors) == 1 ? values[only(predecessors)] :
    tuple((values[id] for id in predecessors)...)

function _graph_output(values, graph::SemanticGraph)
    sinks = graph_sinks(graph)
    length(sinks) == 1 ? values[only(sinks)] : tuple((values[id] for id in sinks)...)
end

"""Validate that every semantic node declares support for `backend`."""
function validate_backend(graph::SemanticGraph, backend::Symbol)
    validate_graph(graph)
    unsupported = [node.id for node in graph.nodes
                   if backend ∉ node_contract(node).backend_compatibility]
    isempty(unsupported) || throw(UnsupportedBackendError(
        "Backend $backend is unsupported by graph nodes $(join(unsupported, ", "))."))
    graph
end

function validate_leakage(graph::SemanticGraph)
    validate_graph(graph)
    for (index, node) in enumerate(graph.nodes)
        node_contract(node).consumes_target && !(node isa PredictorNode) && throw(LeakageError(
            "Graph node $index consumes the target outside a predictor fit operation."))
    end
    graph
end
