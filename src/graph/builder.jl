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

capabilities(chain::Chain) = capabilities(last(chain.steps))

function build_graph(chain::Chain)
    nodes = AbstractGraphNode[]
    for (id, step) in enumerate(chain.steps)
        node = step isa AbstractTransformer ? TransformNode(id, step) :
               step isa AbstractPredictor ? PredictorNode(id, step) :
               throw(GraphValidationError("Chain step $id ($(typeof(step))) is neither a transformer nor predictor."))
        push!(nodes, node)
    end
    SemanticGraph(nodes, [(i, i + 1) for i in 1:length(nodes)-1])
end
