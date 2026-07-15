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
