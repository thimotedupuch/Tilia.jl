fit_graph(graph::SemanticGraph, X, y; context, weights=nothing) =
    fit_graph_backend(context.backend, graph, X, y, context, weights)

function fit_graph_backend(::CPUBackend, graph::SemanticGraph, X, y, context, weights=nothing)
    validate_leakage(graph)
    value = X
    fitted_nodes = AbstractFittedEstimator[]
    node_timings = NamedTuple[]
    for node in graph.nodes
        started = time_ns()
        if node isa TransformNode
            fitted = fit(node.model, value; context=context)
            push!(fitted_nodes, fitted)
            value = transform(fitted, value)
        else
            y === nothing && throw(UnsupportedDataError("Predictor node $(node.id) requires a target."))
            fitted = fit(node.model, value, y; weights=weights, context=context)
            push!(fitted_nodes, fitted)
        end
        push!(node_timings, (node_id=node.id, kind=Symbol(nameof(typeof(node.model))),
                             nanoseconds=time_ns() - started))
    end
    FitReport(observations=size(X, 1), features=size(X, 2),
              details=(nodes=length(graph.nodes), execution=:semantic_graph,
                       weighted=weights !== nothing,
                       node_timings=node_timings)) |>
        report -> FittedGraph(graph, fitted_nodes, report)
end


function fit_graph_backend(backend::AbstractBackend, graph::SemanticGraph, X, y, context, weights=nothing)
    throw(UnsupportedBackendError(
        "No graph execution extension is loaded for backend $(typeof(backend)); load the corresponding optional package or use CPUBackend."))
end

function predict_graph(fitted::FittedGraph, X)
    value = X
    for node in fitted.fitted_nodes
        value = node isa AbstractFittedTransformer ? transform(node, value) : predict(node, value)
    end
    value
end

function predict_proba_graph(fitted::FittedGraph, X)
    value = X
    for node in fitted.fitted_nodes[1:end-1]
        node isa AbstractFittedTransformer || throw(GraphValidationError(
            "Only transformer nodes may precede the final probabilistic predictor."))
        value = transform(node, value)
    end
    isempty(fitted.fitted_nodes) && throw(GraphValidationError("A fitted graph cannot be empty."))
    predict_proba(last(fitted.fitted_nodes), value)
end
