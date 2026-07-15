struct BufferAssignment
    node_id::Int
    buffer_id::Int
    release_after::Int
end

struct ExecutionPlan
    graph::SemanticGraph
    order::Vector{Int}
    buffers::Vector{BufferAssignment}
    peak_buffers::Int
end

"""A lowered numerical operation with explicit execution metadata."""
mutable struct NumericalExecutionNode
    id::Int
    semantic_node_id::Int
    operation::Symbol
    input_shape::Tuple
    output_shape::Tuple
    element_type::Any
    representation::Symbol
    device::Symbol
    buffer_id::Int
    release_after::Int
    aliases::Vector{Int}
    mutability::Symbol
end

"""Backend-neutral lowered fit or inference execution graph."""
struct NumericalExecutionGraph
    phase::Symbol
    nodes::Vector{NumericalExecutionNode}
    edges::Vector{Tuple{Int,Int}}
    peak_buffers::Int
end

_execution_shape(value::AbstractArray) = size(value)
_execution_shape(value::ColumnTable) = size(value)
_execution_shape(value::Tuple) = isempty(value) ? () : (size(first(value), 1), length(value))
_execution_shape(value) = ()

_execution_eltype(value::AbstractArray) = eltype(value)
_execution_eltype(value::ColumnTable) = Any
_execution_eltype(value::Tuple) = isempty(value) ? Any : _execution_eltype(first(value))
_execution_eltype(value) = typeof(value)

_execution_representation(::SparseMatrixCSC) = :sparse
_execution_representation(::AbstractArray) = :dense
_execution_representation(::ColumnTable) = :table
_execution_representation(::Tuple) = :tuple
_execution_representation(::Any) = :scalar

function _record_input!(node::NumericalExecutionNode, value)
    node.input_shape = _execution_shape(value)
    node.element_type = _execution_eltype(value)
    node.representation = _execution_representation(value)
    node
end

function _record_output!(node::NumericalExecutionNode, value)
    node.output_shape = _execution_shape(value)
    node
end

"""Lower a semantic graph to an explicit fit or inference operation graph."""
function _lower_graph(graph::SemanticGraph, input_shape::Tuple, element_type,
                      representation::Symbol; phase::Symbol=:fit,
                      operation::Symbol=:predict, device::Symbol=:cpu)
    phase in (:fit, :inference) || throw(ArgumentError("graph phase must be :fit or :inference."))
    operation in (:predict, :predict_proba) || throw(ArgumentError(
        "inference operation must be :predict or :predict_proba."))
    validate_graph(graph)
    plan = execution_plan(graph)
    assignments = Dict(item.node_id => item for item in plan.buffers)
    nodes = NumericalExecutionNode[]
    for semantic in graph.nodes
        lowered_operation = phase === :fit ?
            (semantic isa TransformNode ? :fit_transform : :fit) :
            (semantic isa TransformNode ? :transform : operation)
        assignment = assignments[semantic.id]
        push!(nodes, NumericalExecutionNode(
            length(nodes) + 1, semantic.id, lowered_operation,
            input_shape, (), element_type, representation,
            device, assignment.buffer_id,
            assignment.release_after, Int[], :owned_output))
    end
    NumericalExecutionGraph(phase, nodes, copy(graph.edges), plan.peak_buffers)
end

function lower_graph(graph::SemanticGraph, input; phase::Symbol=:fit,
                     operation::Symbol=:predict, device::Symbol=:cpu)
    _lower_graph(graph, _execution_shape(input), _execution_eltype(input),
                 _execution_representation(input); phase, operation, device)
end

function lower_graph(graph::SemanticGraph; input_shape::Tuple,
                     element_type=Any, representation::Symbol=:dense,
                     phase::Symbol=:fit, operation::Symbol=:predict,
                     device::Symbol=:cpu)
    _lower_graph(graph, input_shape, element_type, representation;
                 phase, operation, device)
end

"""Plan reusable logical buffers from graph output lifetimes."""
function execution_plan(graph::SemanticGraph)
    validate_graph(graph)
    order = [node.id for node in graph.nodes]
    position = Dict(id => index for (index, id) in enumerate(order))
    last_use = Dict(id => position[id] for id in order)
    for (from, to) in graph.edges
        last_use[from] = max(last_use[from], position[to])
    end
    available = Int[]
    active = Dict{Int,Int}() # node id => buffer id
    assignments = BufferAssignment[]
    next_buffer = 1
    peak = 0
    for (index, id) in enumerate(order)
        for previous in order[1:index-1]
            if haskey(active, previous) && last_use[previous] < index
                push!(available, pop!(active, previous))
            end
        end
        buffer = isempty(available) ? (value = next_buffer; next_buffer += 1; value) : popfirst!(available)
        active[id] = buffer
        push!(assignments, BufferAssignment(id, buffer, last_use[id]))
        peak = max(peak, length(active))
    end
    ExecutionPlan(graph, order, assignments, peak)
end
