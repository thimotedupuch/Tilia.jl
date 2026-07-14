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
    graph
end


function validate_leakage(graph::SemanticGraph)
    validate_graph(graph)
    for (index, node) in enumerate(graph.nodes)
        consumes_target(node) && !(node isa PredictorNode) && throw(LeakageError(
            "Graph node $index consumes the target outside a predictor fit operation."))
    end
    graph
end
