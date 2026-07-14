function _constant_operation(operation, left, right)
    operation === :add && return left + right
    operation === :subtract && return left - right
    operation === :multiply && return left * right
    operation === :divide && return left / right
    throw(GraphValidationError("Unsupported constant operation $operation."))
end

"""Fold pure operations whose inputs are compile-time constants."""
function constant_folding(graph::SemanticGraph)
    validate_graph(graph)
    constants = Dict{Int,Any}()
    nodes = AbstractGraphNode[]
    folded = Set{Int}()
    for node in graph.nodes
        if node isa ConstantNode
            constants[node.id] = node.value
            push!(nodes, node)
        elseif node isa BinaryOperationNode && haskey(constants, node.left) && haskey(constants, node.right)
            value = _constant_operation(node.operation, constants[node.left], constants[node.right])
            constants[node.id] = value
            push!(nodes, ConstantNode(node.id, value))
            push!(folded, node.id)
        else
            push!(nodes, node)
        end
    end
    edges = [edge for edge in graph.edges if !(edge[2] in folded)]
    SemanticGraph(nodes, edges)
end
