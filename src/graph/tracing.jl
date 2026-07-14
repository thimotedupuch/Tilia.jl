struct NodeTrace
    node_id::Int
    operation::Symbol
    nanoseconds::UInt64
    input_shape::Tuple
    output_shape::Tuple
    output_bytes::Int
end

struct ExecutionTrace{T}
    output::T
    nodes::Vector{NodeTrace}
    total_nanoseconds::UInt64
end

_shape(value::AbstractArray) = size(value)
_shape(value::ColumnTable) = size(value)
_shape(value) = ()

"""Execute a fitted graph while collecting per-node timings and output sizes."""
function trace(fitted::FittedGraph, input; operation::Symbol=:predict)
    operation in (:predict, :predict_proba) || throw(ArgumentError(
        "trace operation must be :predict or :predict_proba."))
    value = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    records = NodeTrace[]
    total_started = time_ns()
    for (index, node) in enumerate(fitted.fitted_nodes)
        input_shape = _shape(value)
        started = time_ns()
        node_operation = node isa AbstractFittedTransformer ? :transform :
                         (operation === :predict_proba ? :predict_proba : :predict)
        value = node_operation === :transform ? transform(node, value) :
                node_operation === :predict_proba ? predict_proba(node, value) : predict(node, value)
        elapsed = UInt64(time_ns() - started)
        push!(records, NodeTrace(index, node_operation, elapsed, input_shape,
                                 _shape(value), Base.summarysize(value)))
    end
    ExecutionTrace(value, records, UInt64(time_ns() - total_started))
end

"""Return backend-neutral node and edge data suitable for graph visualization."""
function graph_data(graph::SemanticGraph)
    nodes = [(id=node.id, kind=Symbol(nameof(typeof(node))), learns=learns_state(node),
              consumes_target=consumes_target(node), inference=valid_at_inference(node))
             for node in graph.nodes]
    (nodes=nodes, edges=copy(graph.edges))
end
graph_data(fitted::FittedGraph) = graph_data(fitted.graph)
