function _fuse_standardize(first::FittedStandardize, second::FittedStandardize)
    length(first.means) == length(second.means) || throw(GraphValidationError(
        "Cannot fuse Standardize transforms with different feature counts."))
    means = first.means .+ first.scales .* second.means
    scales = first.scales .* second.scales
    model = Standardize(center=true, scale=true)
    details = hasproperty(second.report.details, :numerical_policy) ?
        (fused=true, original_transforms=2,
         numerical_policy=second.report.details.numerical_policy) :
        (fused=true, original_transforms=2)
    fit_report = FitReport(status=:success, observations=second.report.observations,
        features=length(means), backend=:cpu, details=details,
        root_seed=second.report.root_seed, stream_id=second.report.stream_id,
        deterministic=second.report.deterministic,
        thread_count=second.report.thread_count)
    FittedStandardize(model, means, scales, fit_report, second.schema)
end

"""Fuse adjacent fitted affine standardizations for inference."""
function transform_fusion(fitted::FittedGraph)
    _is_linear_graph(fitted.graph) || return fitted
    nodes = AbstractFittedEstimator[]
    fused_count = 0
    index = 1
    while index <= length(fitted.fitted_nodes)
        if fitted.fitted_nodes[index] isa FittedStandardize
            current = fitted.fitted_nodes[index]
            index += 1
            while index <= length(fitted.fitted_nodes) &&
                  fitted.fitted_nodes[index] isa FittedStandardize
                current = _fuse_standardize(current, fitted.fitted_nodes[index])
                fused_count += 1
                index += 1
            end
            push!(nodes, current)
        else
            push!(nodes, fitted.fitted_nodes[index])
            index += 1
        end
    end
    models = tuple((node.model for node in nodes)...)
    graph = validate_graph(build_graph(Chain(models...)))
    has_execution_metadata = hasproperty(fitted.report.details, :fit_execution_graph) &&
                             hasproperty(fitted.report.details, :inference_execution_graph)
    original_fit = has_execution_metadata ? fitted.report.details.fit_execution_graph : nothing
    original_inference = has_execution_metadata ? fitted.report.details.inference_execution_graph : nothing
    input_shape = has_execution_metadata ? first(original_fit.nodes).input_shape :
        (fitted.report.observations, fitted.report.features)
    element_type = has_execution_metadata ? first(original_fit.nodes).element_type : Any
    representation = has_execution_metadata ? first(original_fit.nodes).representation : :dense
    device = has_execution_metadata ? first(original_fit.nodes).device : :cpu
    fit_execution = lower_graph(graph; input_shape, element_type,
        representation, phase=:fit, device)
    inference_execution = lower_graph(graph; input_shape, element_type,
        representation, phase=:inference, device)
    for index in eachindex(nodes)
        fit_execution.nodes[index].input_shape = input_shape
        inference_execution.nodes[index].input_shape = input_shape
        if nodes[index] isa AbstractFittedTransformer
            fit_execution.nodes[index].output_shape = input_shape
            inference_execution.nodes[index].output_shape = input_shape
        else
            inference_execution.nodes[index].output_shape =
                has_execution_metadata ? last(original_inference.nodes).output_shape :
                (fitted.report.observations,)
        end
    end
    propagated = hasproperty(fitted.report.details, :input_schema) ?
        propagate_schema(graph, fitted.report.details.input_schema;
                         observations=fitted.report.observations) : nothing
    schema_details = propagated === nothing ? (;) : (propagated_schemas=propagated,)
    details = merge(fitted.report.details, schema_details,
                    (fit_execution_graph=fit_execution,
                     inference_execution_graph=inference_execution,
                     optimization=(fused_transforms=fused_count,
                                   original_nodes=length(fitted.fitted_nodes),
                                   optimized_nodes=length(nodes)),))
    fit_report = FitReport(status=fitted.report.status, observations=fitted.report.observations,
        features=fitted.report.features, backend=fitted.report.backend,
        warnings=copy(fitted.report.warnings), details=details,
        root_seed=fitted.report.root_seed, stream_id=fitted.report.stream_id,
        deterministic=fitted.report.deterministic,
        thread_count=fitted.report.thread_count)
    FittedGraph(graph, nodes, fit_report)
end
