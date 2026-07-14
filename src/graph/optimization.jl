"""Apply semantics-preserving inference optimizations to a fitted graph."""
optimize(fitted::FittedGraph) = transform_fusion(fitted)

"""Run structural semantic-graph cleanup passes."""
function optimize(graph::SemanticGraph; outputs=[last(graph.nodes).id])
    folded = constant_folding(graph)
    converted = redundant_conversion_elimination(folded)
    mapped_outputs = outputs == [last(graph.nodes).id] ? [last(converted.nodes).id] : outputs
    dead_node_elimination(converted; outputs=mapped_outputs)
end
