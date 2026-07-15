mutable struct FittedReactantGraph{F,C,B} <: Tilia.AbstractFittedEstimator
    cpu_graph::F
    cache::C
    backend::B
    compilation_nanoseconds::UInt64
    cache_hits::Int
    last_execution_nanoseconds::UInt64
    transferred_bytes::Int
    accelerator_nodes::Vector{Int}
    host_fit_nodes::Vector{Int}
    accelerated_objective_value::Float64
end

function Tilia.fit_graph_backend(backend::Tilia.ReactantBackend,
                                 graph::Tilia.SemanticGraph, X, y, context, weights=nothing)
    Tilia.validate_leakage(graph)
    supported, reason = _supported(graph, X)
    cpu_graph = Tilia.fit_graph_backend(Tilia.CPUBackend(), graph, X, y,
                                        _cpu_context(context), weights)
    if !supported
        backend.fallback === :cpu && return _fallback_graph(cpu_graph, reason)
        throw(Tilia.UnsupportedBackendError(
            "Reactant cannot execute this graph: $reason. Use ReactantBackend(fallback=:cpu) for an explicit host fallback."))
    end
    fitted = FittedReactantGraph(cpu_graph, context.cache, backend, UInt64(0), 0,
                                 UInt64(0), 0, collect(1:length(graph.nodes)),
                                 collect(1:length(graph.nodes)), NaN)
    try
        backend.device !== :auto && Reactant.set_default_backend(string(backend.device))
        _, compilation_time, transferred = _compile_for!(fitted, X)
        fitted.compilation_nanoseconds += compilation_time
        fitted.transferred_bytes += transferred
        objective, objective_time, objective_transferred = _compile_objective(cpu_graph, X, y, weights)
        fitted.accelerated_objective_value = objective
        fitted.compilation_nanoseconds += objective_time
        fitted.transferred_bytes += objective_transferred
    catch error
        backend.fallback === :cpu && return _fallback_graph(
            cpu_graph, "Reactant compilation failed: $(sprint(showerror, error))")
        throw(Tilia.UnsupportedBackendError(
            "Reactant compilation failed for the supported graph: $(sprint(showerror, error)). Use ReactantBackend(fallback=:cpu) for explicit fallback."))
    end
    fitted
end

function Tilia.predict_proba(fitted::FittedReactantGraph, X::AbstractMatrix)
    standardize, _ = _arrays(fitted)
    size(X, 2) == length(standardize.means) || throw(Tilia.SchemaMismatchError(
        "Reactant graph was fitted with $(length(standardize.means)) features; received $(size(X, 2))."))
    entry, compilation_time, transferred = _compile_for!(fitted, X)
    fitted.compilation_nanoseconds += compilation_time
    device_X = Reactant.to_rarray(Matrix(X))
    started = time_ns()
    device_result = entry.compiled(device_X, entry.parameters...)
    result = Array(device_result)
    fitted.last_execution_nanoseconds = UInt64(time_ns() - started)
    fitted.transferred_bytes += transferred + Base.summarysize(X) + Base.summarysize(result)
    result
end

function Tilia.predict(fitted::FittedReactantGraph, X::AbstractMatrix)
    probabilities = Tilia.predict_proba(fitted, X)
    classes = fitted.cpu_graph.fitted_nodes[2].classes
    [classes[argmax(view(probabilities, row, :))] for row in axes(probabilities, 1)]
end
