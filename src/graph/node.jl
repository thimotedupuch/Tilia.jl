abstract type AbstractGraphNode end

"""Complete semantic contract declared by a graph node."""
struct NodeContract
    input::NamedTuple
    output_schema_rule::Symbol
    learns_state::Bool
    consumes_target::Bool
    changes_row_count::Bool
    changes_feature_count::Bool
    valid_at_inference::Bool
    sparse_compatible::Bool
    missing_compatible::Bool
    backend_compatibility::Tuple{Vararg{Symbol}}
end

struct TransformNode{M<:AbstractEstimator} <: AbstractGraphNode
    id::Int
    model::M
end

struct PredictorNode{M<:AbstractEstimator} <: AbstractGraphNode
    id::Int
    model::M
end

"""Explicit representation conversion used during graph lowering."""
struct ConversionNode <: AbstractGraphNode
    id::Int
    from::Symbol
    to::Symbol
end

"""Compile-time scalar or array value in a numerical graph."""
struct ConstantNode{T} <: AbstractGraphNode
    id::Int
    value::T
end

"""Pure binary numerical operation eligible for constant folding."""
struct BinaryOperationNode <: AbstractGraphNode
    id::Int
    operation::Symbol
    left::Int
    right::Int
    function BinaryOperationNode(id::Int, operation::Symbol, left::Int, right::Int)
        operation in (:add, :subtract, :multiply, :divide) || throw(GraphValidationError(
            "Unsupported pure binary graph operation $operation."))
        new(id, operation, left, right)
    end
end

learns_state(::AbstractGraphNode) = true
consumes_target(::TransformNode) = false
consumes_target(node::PredictorNode) =
    capabilities(node.model).task in (:classification, :regression)
valid_at_inference(::AbstractGraphNode) = true
learns_state(::ConversionNode) = false
consumes_target(::ConversionNode) = false
learns_state(::Union{ConstantNode,BinaryOperationNode}) = false
consumes_target(::Union{ConstantNode,BinaryOperationNode}) = false

preserves_feature_count(::AbstractEstimator) = false
backend_compatibility(::AbstractEstimator) = (:cpu,)

function node_contract(node::Union{TransformNode,PredictorNode})
    declared = capabilities(node.model)
    NodeContract(
        input_contract(node.model),
        :model_dispatch,
        learns_state(node),
        consumes_target(node),
        false,
        node isa PredictorNode || !preserves_feature_count(node.model),
        valid_at_inference(node),
        declared.sparse,
        declared.missing,
        backend_compatibility(node.model),
    )
end

function node_contract(node::ConversionNode)
    NodeContract((; rows_are_observations=true, representation=node.from),
        :representation_conversion, false, false, false, false, true,
        node.from === :sparse || node.to === :sparse, false, (:cpu, :reactant))
end

function node_contract(node::ConstantNode)
    NodeContract((; constant=true), :constant, false, false, false, false, true,
        node.value isa SparseMatrixCSC, false, (:cpu, :reactant))
end

function node_contract(::BinaryOperationNode)
    NodeContract((; arity=2), :elementwise_binary, false, false, false, false, true,
        true, false, (:cpu, :reactant))
end
