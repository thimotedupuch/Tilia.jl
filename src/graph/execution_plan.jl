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
