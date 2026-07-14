"""Eliminate identity representation conversions while reconnecting their users."""
function redundant_conversion_elimination(graph::SemanticGraph)
    validate_graph(graph)
    redundant = Set(node.id for node in graph.nodes
                    if node isa ConversionNode && node.from === node.to)
    isempty(redundant) && return graph
    edges = copy(graph.edges)
    for id in sort!(collect(redundant))
        predecessors = [from for (from, to) in edges if to == id]
        successors = [to for (from, to) in edges if from == id]
        filter!(edge -> edge[1] != id && edge[2] != id, edges)
        append!(edges, [(from, to) for from in predecessors for to in successors])
    end
    kept = [node for node in graph.nodes if !(node.id in redundant)]
    mapping = Dict(node.id => index for (index, node) in enumerate(kept))
    nodes = AbstractGraphNode[_reid(node, mapping[node.id], mapping) for node in kept]
    remapped_edges = [(mapping[from], mapping[to]) for (from, to) in edges
                      if haskey(mapping, from) && haskey(mapping, to) && mapping[from] != mapping[to]]
    SemanticGraph(nodes, unique(remapped_edges))
end
