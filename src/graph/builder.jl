"""A sequential composition executed as a semantic graph."""
struct Chain{S<:Tuple} <: AbstractEstimator
    steps::S
    function Chain(steps::AbstractEstimator...)
        isempty(steps) && throw(InvalidHyperparameterError("Chain requires at least one estimator."))
        new{typeof(steps)}(steps)
    end
end

struct Parallel{S<:Tuple} <: AbstractTransformer
    steps::S
    function Parallel(steps::AbstractTransformer...)
        isempty(steps) && throw(InvalidHyperparameterError("Parallel requires at least one transformer."))
        new{typeof(steps)}(steps)
    end
end

struct ColumnMap{S<:Tuple} <: AbstractTransformer
    mappings::S
    function ColumnMap(mappings::Pair...)
        isempty(mappings) && throw(InvalidHyperparameterError("ColumnMap requires at least one mapping."))
        all(mapping -> last(mapping) isa AbstractTransformer, mappings) ||
            throw(InvalidHyperparameterError("Every ColumnMap value must be a transformer."))
        new{typeof(mappings)}(mappings)
    end
end


struct Select{S} <: AbstractTransformer
    columns::S
end
Select(first, second, rest...) = Select((first, second, rest...))

struct Concatenate <: AbstractTransformer end

function capabilities(chain::Chain)
    final = capabilities(last(chain.steps))
    input_declaration = capabilities(first(chain.steps))
    merge(final, (
        sparse=input_declaration.sparse,
        missing=input_declaration.missing,
        partial_fit=false,
    ))
end

function _append_node!(nodes, edges, step, predecessors)
    id = length(nodes) + 1
    task = capabilities(step).task
    node = step isa AbstractTransformer || task in (:transformation, :neighbors) ?
           TransformNode(id, step) : PredictorNode(id, step)
    push!(nodes, node)
    append!(edges, ((from, id) for from in predecessors))
    [id]
end

function _append_step!(nodes, edges, step, predecessors)
    if step isa Parallel
        outputs = Int[]
        for branch in step.steps
            append!(outputs, _append_step!(nodes, edges, branch, predecessors))
        end
        return outputs
    elseif step isa ColumnMap
        outputs = Int[]
        for mapping in step.mappings
            selected = _append_node!(nodes, edges, Select(first(mapping)), predecessors)
            append!(outputs, _append_step!(nodes, edges, last(mapping), selected))
        end
        return length(outputs) == 1 ? outputs :
               _append_node!(nodes, edges, Concatenate(), outputs)
    end
    _append_node!(nodes, edges, step, predecessors)
end

function build_graph(chain::Chain)
    nodes = AbstractGraphNode[]
    edges = Tuple{Int,Int}[]
    predecessors = Int[]
    for step in chain.steps
        predecessors = _append_step!(nodes, edges, step, predecessors)
    end
    SemanticGraph(nodes, edges)
end
