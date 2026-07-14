function _fuse_standardize(first::FittedStandardize, second::FittedStandardize)
    length(first.means) == length(second.means) || throw(GraphValidationError(
        "Cannot fuse Standardize transforms with different feature counts."))
    means = first.means .+ first.scales .* second.means
    scales = first.scales .* second.scales
    model = Standardize(center=true, scale=true)
    details = (fused=true, original_transforms=2)
    fit_report = FitReport(status=:success, observations=second.report.observations,
        features=length(means), backend=:cpu, details=details)
    FittedStandardize(model, means, scales, fit_report, second.schema)
end

"""Fuse adjacent fitted affine standardizations for inference."""
function transform_fusion(fitted::FittedGraph)
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
    details = merge(fitted.report.details,
                    (optimization=(fused_transforms=fused_count,
                                   original_nodes=length(fitted.fitted_nodes),
                                   optimized_nodes=length(nodes)),))
    fit_report = FitReport(status=fitted.report.status, observations=fitted.report.observations,
        features=fitted.report.features, backend=fitted.report.backend,
        warnings=copy(fitted.report.warnings), details=details)
    FittedGraph(graph, nodes, fit_report)
end
