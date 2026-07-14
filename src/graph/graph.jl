struct SemanticGraph
    nodes::Vector{AbstractGraphNode}
    edges::Vector{Tuple{Int,Int}}
end

struct FittedGraph <: AbstractFittedEstimator
    graph::SemanticGraph
    fitted_nodes::Vector{AbstractFittedEstimator}
    report::FitReport
end
