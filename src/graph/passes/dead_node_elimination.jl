_reid(node::TransformNode, id) = TransformNode(id, node.model)
_reid(node::PredictorNode, id) = PredictorNode(id, node.model)
_reid(node::ConversionNode, id) = ConversionNode(id, node.from, node.to)
_reid(node::ConstantNode, id) = ConstantNode(id, node.value)
_reid(node::BinaryOperationNode, id) = BinaryOperationNode(id, node.operation,
    node.left, node.right)
_reid(node, id, mapping) = _reid(node, id)
_reid(node::BinaryOperationNode, id, mapping) = BinaryOperationNode(id, node.operation,
    mapping[node.left], mapping[node.right])

"""Remove nodes that cannot reach any requested graph output."""
function dead_node_elimination(graph::SemanticGraph; outputs=[last(graph.nodes).id])
    validate_graph(graph)
    live = Set(Int.(outputs))
    changed = true
    while changed
        changed = false
        for (from, to) in graph.edges
            if to in live && !(from in live)
                push!(live, from)
                changed = true
            end
        end
    end
    all(id -> 1 <= id <= length(graph.nodes), live) || throw(GraphValidationError(
        "dead-node elimination requested an unknown output node."))
    kept = [node for node in graph.nodes if node.id in live]
    mapping = Dict(node.id => index for (index, node) in enumerate(kept))
    nodes = AbstractGraphNode[_reid(node, mapping[node.id], mapping) for node in kept]
    edges = [(mapping[from], mapping[to]) for (from, to) in graph.edges
             if haskey(mapping, from) && haskey(mapping, to)]
    SemanticGraph(nodes, unique(edges))
end
