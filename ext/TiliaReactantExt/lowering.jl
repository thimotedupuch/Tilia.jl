function _supported(graph, X)
    X isa AbstractMatrix || return false, "Reactant prototype accepts dense matrices only"
    length(graph.nodes) == 2 || return false, "Reactant prototype requires exactly two graph nodes"
    graph.nodes[1] isa Tilia.TransformNode && graph.nodes[1].model isa Tilia.Standardize ||
        return false, "first node must be Standardize"
    graph.nodes[2] isa Tilia.PredictorNode && graph.nodes[2].model isa Tilia.LogisticRegression ||
        return false, "final node must be LogisticRegression"
    eltype(X) in (Float32, Float64) || return false, "Reactant prototype supports Float32 and Float64"
    true, ""
end

function _cpu_context(context)
    Tilia.FitContext(backend=Tilia.CPUBackend(), rng=context.rng,
        numerics=context.numerics, deterministic=context.deterministic,
        cache=context.cache, root_seed=context.root_seed,
        stream_id=context.stream_id)
end
