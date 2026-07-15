module TiliaReactantExt

using Tilia
using Reactant

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

function _reactant_probabilities(X, means, scales, coefficients, intercept)
    standardized = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)
    scores = standardized * coefficients .+ reshape(intercept, 1, :)
    positive = one(eltype(scores)) ./ (one(eltype(scores)) .+ exp.(-scores))
    if size(coefficients, 2) == 1
        return cat(one(eltype(positive)) .- positive, positive; dims=2)
    end
    positive ./ sum(positive; dims=2)
end

function _reactant_logistic_objective(X, target, weights, means, scales,
                                      coefficients, intercept, lambda)
    standardized = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)
    scores = standardized * coefficients .+ reshape(intercept, 1, :)
    losses = max.(scores, zero(eltype(scores))) .- scores .* target .+
             log1p.(exp.(-abs.(scores)))
    sum(losses .* weights) + sum(lambda) / 2 * sum(abs2, coefficients)
end

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


function _compile_objective(cpu_graph, X, y, observation_weights=nothing)
    standardize = cpu_graph.fitted_nodes[1]
    logistic = cpu_graph.fitted_nodes[2]
    T = eltype(logistic.coefficients)
    trained_classes = length(logistic.classes) == 2 ? logistic.classes[end:end] : logistic.classes
    target = hcat((T.(y .== class) for class in trained_classes)...)
    weights = observation_weights === nothing ? ones(T, length(y), 1) :
              reshape(T.(observation_weights), :, 1)
    lambda = T[logistic.model.lambda]
    host_arrays = (Matrix{T}(X), target, weights, standardize.means,
                   standardize.scales, logistic.coefficients, logistic.intercept, lambda)
    transferred = sum(Base.summarysize, host_arrays)
    device_arrays = map(Reactant.to_rarray, host_arrays)
    started = time_ns()
    compiled = Reactant.compile(_reactant_logistic_objective, device_arrays)
    result = compiled(device_arrays...)
    elapsed = UInt64(time_ns() - started)
    Reactant.to_number(result), elapsed, transferred
end

function _cpu_context(context)
    Tilia.FitContext(backend=Tilia.CPUBackend(), rng=context.rng,
        numerics=context.numerics, deterministic=context.deterministic,
        cache=context.cache, root_seed=context.root_seed,
        stream_id=context.stream_id)
end

function _fallback_graph(cpu_graph, reason)
    warning = "Reactant fallback to CPU: $reason"
    details = merge(cpu_graph.report.details, (
        requested_backend=:reactant,
        accelerator_nodes=Int[], host_nodes=collect(1:length(cpu_graph.graph.nodes)),
        transfer_locations=NamedTuple[], compilation_nanoseconds=UInt64(0),
        execution_nanoseconds=UInt64(0), transferred_bytes=0,
        unsupported_operations=[reason], fallback_operations=[warning],
    ))
    fit_report = Tilia.FitReport(status=cpu_graph.report.status,
        observations=cpu_graph.report.observations, features=cpu_graph.report.features,
        backend=:cpu, warnings=[cpu_graph.report.warnings; warning], details=details,
        root_seed=cpu_graph.report.root_seed, stream_id=cpu_graph.report.stream_id,
        deterministic=cpu_graph.report.deterministic,
        thread_count=cpu_graph.report.thread_count)
    Tilia.FittedGraph(cpu_graph.graph, cpu_graph.fitted_nodes, fit_report)
end

function _arrays(fitted::FittedReactantGraph)
    standardize = fitted.cpu_graph.fitted_nodes[1]
    logistic = fitted.cpu_graph.fitted_nodes[2]
    standardize, logistic
end

function _compile_for!(fitted::FittedReactantGraph, X::AbstractMatrix)
    standardize, logistic = _arrays(fitted)
    key = UInt64(hash((:tilia_reactant_probability_v1, eltype(X), size(X),
                       length(standardize.means), size(logistic.coefficients))))
    lock(fitted.cache.lock) do
        if haskey(fitted.cache.entries, key)
            fitted.cache_hits += 1
            return fitted.cache.entries[key], UInt64(0), 0
        end
        host_arrays = (Matrix(X), standardize.means, standardize.scales,
                       logistic.coefficients, logistic.intercept)
        transferred = sum(Base.summarysize, host_arrays)
        device_arrays = map(Reactant.to_rarray, host_arrays)
        started = time_ns()
        compiled = Reactant.compile(_reactant_probabilities, device_arrays)
        elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled, parameters=device_arrays[2:end])
        fitted.cache.entries[key] = entry
        entry, elapsed, transferred
    end
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

function Tilia.report(fitted::FittedReactantGraph)
    base = fitted.cpu_graph.report
    device = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    details = merge(base.details, (
        requested_backend=:reactant, device=device,
        accelerator_nodes=copy(fitted.accelerator_nodes),
        host_nodes=Int[], host_fit_nodes=copy(fitted.host_fit_nodes),
        transfer_locations=[(location=:input, direction=:host_to_device),
                            (location=:output, direction=:device_to_host)],
        compilation_nanoseconds=fitted.compilation_nanoseconds,
        execution_nanoseconds=fitted.last_execution_nanoseconds,
        transferred_bytes=fitted.transferred_bytes,
        unsupported_operations=String[],
        fallback_operations=["fit-time statistics and Newton solver executed on CPU"],
        compilation_cache_hits=fitted.cache_hits,
        accelerated_logistic_objective=true,
        accelerated_objective_value=fitted.accelerated_objective_value,
    ))
    Tilia.FitReport(status=base.status, observations=base.observations,
        features=base.features, backend=:reactant,
        warnings=[base.warnings; "Fit-time statistics and Newton solver used an explicit CPU path."],
        details=details, root_seed=base.root_seed, stream_id=base.stream_id,
        deterministic=base.deterministic, thread_count=base.thread_count)
end

Tilia.save_model(path::AbstractString, fitted::FittedReactantGraph) =
    Tilia.save_model(path, fitted.cpu_graph)

end
