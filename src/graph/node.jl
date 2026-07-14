abstract type AbstractGraphNode end

struct TransformNode{M<:AbstractTransformer} <: AbstractGraphNode
    id::Int
    model::M
end

struct PredictorNode{M<:AbstractPredictor} <: AbstractGraphNode
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
consumes_target(::PredictorNode) = true
valid_at_inference(::AbstractGraphNode) = true
learns_state(::ConversionNode) = false
consumes_target(::ConversionNode) = false
learns_state(::Union{ConstantNode,BinaryOperationNode}) = false
consumes_target(::Union{ConstantNode,BinaryOperationNode}) = false
