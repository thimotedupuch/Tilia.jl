function _fallback_graph(cpu_graph, reason)
    warning = "Reactant fallback to CPU: $reason"
    details = merge(cpu_graph.report.details, (
        requested_backend=:reactant,
        accelerator_nodes=Int[], host_nodes=collect(1:length(cpu_graph.graph.nodes)),
        transfer_locations=NamedTuple[], compilation_nanoseconds=UInt64(0),
        execution_nanoseconds=UInt64(0), transferred_bytes=0,
        phase_placement=(fit=[(node_id=id, device=:cpu) for id in eachindex(cpu_graph.graph.nodes)],
                         inference=[(node_id=id, device=:cpu) for id in eachindex(cpu_graph.graph.nodes)]),
        phase_timings=(compilation_nanoseconds=UInt64(0),
                       host_conversion_nanoseconds=UInt64(0),
                       fit_objective_device_execution_nanoseconds=UInt64(0),
                       fit_objective_synchronization_and_materialization_nanoseconds=UInt64(0),
                       inference_device_execution_nanoseconds=UInt64(0),
                       inference_synchronization_and_materialization_nanoseconds=UInt64(0)),
        transfer_accounting=(kind=:estimated_host_summarysize, estimated_bytes=0,
                             last_result_estimated_bytes=0,
                             actual_device_transfer_bytes=missing),
        memory_accounting=(kind=:estimated_host_summarysize,
                           portable_model_host_bytes=Base.summarysize(cpu_graph),
                           compilation_cache_host_bytes=0,
                           retained_device_parameter_bytes=0,
                           retained_executable_device_bytes=missing,
                           actual_peak_host_bytes=missing,
                           actual_peak_device_bytes=missing),
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
    cache_snapshot = Tilia._compilation_cache_snapshot(fitted.cache)
    node_count = length(fitted.cpu_graph.graph.nodes)
    accelerator_ids = Set(fitted.accelerator_nodes)
    host_ids = [id for id in 1:node_count if id ∉ accelerator_ids]
    boundary_transfers = NamedTuple[]
    for (from, to) in fitted.cpu_graph.graph.edges
        from_device = from in accelerator_ids ? :reactant : :cpu
        to_device = to in accelerator_ids ? :reactant : :cpu
        from_device == to_device && continue
        push!(boundary_transfers, (
            location=:region_boundary,
            direction=from_device === :cpu ? :host_to_device : :device_to_host,
            from_node=from, to_node=to))
    end
    details = merge(base.details, (
        requested_backend=:reactant, device=fitted.resolved_device,
        accelerator_nodes=copy(fitted.accelerator_nodes),
        host_nodes=host_ids,
        host_fit_nodes=copy(fitted.host_fit_nodes),
        accelerator_fit_nodes=copy(fitted.accelerator_fit_nodes),
        transfer_locations=vcat(
            NamedTuple[(location=:input,
                        direction=fitted.last_input_location === :host ? :host_to_device : :device_resident),
                       (location=:output,
                        direction=fitted.last_output_location === :host ? :device_to_host : :device_resident),
                       (location=:class_indices, direction=:device_to_host,
                        note="predict transfers indices rather than probabilities")],
            boundary_transfers),
        compilation_nanoseconds=fitted.compilation_nanoseconds,
        execution_nanoseconds=fitted.last_execution_nanoseconds,
        transferred_bytes=fitted.transferred_bytes,
        phase_placement=(
            fit=[(node_id=id,
                  device=id in fitted.accelerator_fit_nodes ? :reactant : :cpu)
                 for id in 1:node_count],
            inference=[(node_id=id,
                        device=id in accelerator_ids ? :reactant : :cpu)
                       for id in 1:node_count]),
        phase_timings=(
            compilation_nanoseconds=fitted.compilation_nanoseconds,
            host_conversion_nanoseconds=fitted.host_conversion_nanoseconds,
            fit_objective_device_execution_nanoseconds=fitted.fit_device_execution_nanoseconds,
            fit_objective_synchronization_and_materialization_nanoseconds=fitted.fit_materialization_nanoseconds,
            fit_statistics_device_execution_nanoseconds=fitted.fit_statistics_device_execution_nanoseconds,
            fit_statistics_synchronization_and_materialization_nanoseconds=fitted.fit_statistics_materialization_nanoseconds,
            fit_optimizer_device_execution_nanoseconds=fitted.fit_optimizer_device_execution_nanoseconds,
            fit_optimizer_synchronization_and_materialization_nanoseconds=fitted.fit_optimizer_materialization_nanoseconds,
            inference_device_execution_nanoseconds=fitted.last_execution_nanoseconds,
            inference_synchronization_and_materialization_nanoseconds=fitted.last_materialization_nanoseconds),
        transfer_accounting=(kind=:estimated_host_summarysize,
                             estimated_bytes=fitted.transferred_bytes,
                             last_result_estimated_bytes=fitted.last_result_estimated_bytes,
                             last_input_location=fitted.last_input_location,
                             last_output_location=fitted.last_output_location,
                             actual_device_transfer_bytes=missing),
        memory_accounting=(kind=:estimated_host_summarysize,
                           portable_model_host_bytes=Base.summarysize(fitted.cpu_graph),
                           compilation_cache_host_bytes=cache_snapshot.retained_host_bytes,
                           retained_device_parameter_bytes=0,
                           retained_executable_device_bytes=missing,
                           actual_peak_host_bytes=missing,
                           actual_peak_device_bytes=missing),
        unsupported_operations=copy(fitted.unsupported_operations),
        fallback_operations=isempty(fitted.host_fit_nodes) ? String[] :
            ["remaining fit-time transforms and model solvers executed on CPU"],
        compilation_cache_hits=fitted.cache_hits,
        compilation_count=cache_snapshot.compilations,
        compilation_cache_size=cache_snapshot.size,
        compilation_cache_capacity=cache_snapshot.capacity,
        compilation_cache_evictions=cache_snapshot.evictions,
        accelerated_logistic_objective=isfinite(fitted.accelerated_objective_value),
        accelerated_objective_value=fitted.accelerated_objective_value,
        reactant_capabilities=fitted.capabilities,
    ))
    Tilia.FitReport(status=base.status, observations=base.observations,
        features=base.features, backend=:reactant,
        warnings=isempty(fitted.host_fit_nodes) ? base.warnings :
            [base.warnings; "Remaining fit-time transforms and model solvers used an explicit CPU path."],
        details=details, root_seed=base.root_seed, stream_id=base.stream_id,
        deterministic=base.deterministic, thread_count=base.thread_count)
end

Tilia.save_model(path::AbstractString, fitted::FittedReactantGraph) =
    Tilia.save_model(path, fitted.cpu_graph)
