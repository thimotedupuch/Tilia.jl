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
    primitives::Vector{NumericalExecutionNode}
    primitive_edges::Vector{Tuple{Int,Int}}
end

# Positional compatibility for version-2 structural persistence payloads.
NumericalExecutionGraph(phase::Symbol, nodes::Vector{NumericalExecutionNode},
                        edges::Vector{Tuple{Int,Int}}, peak_buffers::Int) =
    NumericalExecutionGraph(phase, nodes, edges, peak_buffers,
                            NumericalExecutionNode[], Tuple{Int,Int}[])

function _primitive_operations(model, phase::Symbol, inference_operation::Symbol)
    if model isa Standardize
        return phase === :fit ? [:reduce_mean, :center, :reduce_variance, :normalize] : [:normalize]
    elseif model isa Impute
        return phase === :fit ? [:missing_mask, :reduce_statistic, :select_fill] :
                                [:missing_mask, :select_fill]
    elseif model isa OneHotEncode
        return [:categorical_lookup, :gather, :scatter]
    elseif model isa Union{Select,ColumnMap}
        return [:gather]
    elseif model isa Parallel
        return [:fork]
    elseif model isa Concatenate
        return [:concatenate]
    elseif model isa Union{PCA,TruncatedSVD}
        return phase === :fit ? [:center, :svd, :select_components] : [:center, :matmul]
    elseif model isa MeanRegressor
        return phase === :fit ? [:weighted_reduction] : [:fill]
    elseif model isa Union{LinearRegression,RidgeRegression}
        if phase === :fit
            solver = model.solver
            factorization = solver === :qr ? :qr_factorization :
                            solver === :svd ? :svd : :cholesky_factorization
            return [:center, factorization, :triangular_solve]
        end
        return [:matmul, :add]
    elseif model isa Union{LogisticRegression,SparseLogisticRegression}
        return phase === :fit ? [:matmul, :sigmoid, :weighted_reduction, :solver_loop] :
                                [:matmul, :add, :sigmoid, :normalize]
    elseif model isa Union{Lasso,ElasticNet}
        return phase === :fit ? [:center, :coordinate_descent_loop] : [:matmul, :add]
    elseif model isa AbstractTransformer
        return phase === :fit ? [:solver_loop, :transform] : [:transform]
    elseif phase === :fit
        return [:solver_loop]
    elseif capabilities(model).probabilistic && inference_operation === :predict_proba
        return [:score, :stable_probability_normalization]
    end
    [:score, :select_output]
end

function _lower_primitives(graph::SemanticGraph, nodes::Vector{NumericalExecutionNode},
                           phase::Symbol, inference_operation::Symbol)
    primitives = NumericalExecutionNode[]
    edges = Tuple{Int,Int}[]
    previous_group_last = 0
    for (semantic, region) in zip(graph.nodes, nodes)
        operations = semantic isa Union{TransformNode,PredictorNode} ?
            _primitive_operations(semantic.model, phase, inference_operation) : [region.operation]
        group_first = length(primitives) + 1
        for operation in operations
            primitive_id = length(primitives) + 1
            push!(primitives, NumericalExecutionNode(
                primitive_id, semantic.id, operation, region.input_shape, (),
                region.element_type, region.representation, region.device,
                region.buffer_id, region.release_after, Int[], :owned_output))
            primitive_id > group_first && push!(edges, (primitive_id - 1, primitive_id))
        end
        previous_group_last > 0 && push!(edges, (previous_group_last, group_first))
        previous_group_last = length(primitives)
    end
    primitives, edges
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

function _record_primitive_region!(graph::NumericalExecutionGraph,
                                   semantic_node_id::Int, input, output)
    indices = findall(node -> node.semantic_node_id == semantic_node_id,
                      graph.primitives)
    isempty(indices) && return graph
    current_shape = _execution_shape(input)
    current_type = _execution_eltype(input)
    current_representation = _execution_representation(input)
    for (position, index) in enumerate(indices)
        primitive = graph.primitives[index]
        primitive.input_shape = current_shape
        primitive.element_type = current_type
        primitive.representation = current_representation
        if position == length(indices)
            primitive.output_shape = _execution_shape(output)
            current_shape = primitive.output_shape
            current_type = _execution_eltype(output)
            current_representation = _execution_representation(output)
        else
            primitive.output_shape = current_shape
        end
    end
    graph
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
    primitives, primitive_edges = _lower_primitives(graph, nodes, phase, operation)
    NumericalExecutionGraph(phase, nodes, copy(graph.edges), plan.peak_buffers,
                            primitives, primitive_edges)
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
