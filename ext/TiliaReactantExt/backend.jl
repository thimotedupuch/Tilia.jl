mutable struct FittedReactantGraph{F,C,B,N,G} <: Tilia.AbstractFittedEstimator
    cpu_graph::F
    cache::C
    backend::B
    numerical_policy::N
    capabilities::G
    execution_lock::ReentrantLock
    resolved_device::Symbol
    inference_kind::Symbol
    region_start::Int
    regions::Vector{UnitRange{Int}}
    unsupported_operations::Vector{String}
    compilation_nanoseconds::UInt64
    cache_hits::Int
    host_conversion_nanoseconds::UInt64
    fit_device_execution_nanoseconds::UInt64
    fit_materialization_nanoseconds::UInt64
    fit_statistics_device_execution_nanoseconds::UInt64
    fit_statistics_materialization_nanoseconds::UInt64
    fit_optimizer_device_execution_nanoseconds::UInt64
    fit_optimizer_materialization_nanoseconds::UInt64
    last_execution_nanoseconds::UInt64
    last_materialization_nanoseconds::UInt64
    last_result_estimated_bytes::Int
    transferred_bytes::Int
    last_input_location::Symbol
    last_output_location::Symbol
    accelerator_nodes::Vector{Int}
    host_fit_nodes::Vector{Int}
    accelerator_fit_nodes::Vector{Int}
    accelerated_objective_value::Float64
end

function Tilia.fit_graph_backend(backend::Tilia.ReactantBackend,
                                 graph::Tilia.SemanticGraph, X, y, context, weights=nothing)
    Tilia.validate_leakage(graph)
    semantic_models = [node.model for node in graph.nodes
                       if node isa Union{Tilia.TransformNode,Tilia.PredictorNode}]
    head_model = isempty(semantic_models) ? nothing : last(semantic_models)
    inference_kind = _probabilistic_head(head_model) ? :probabilities : :regression
    inference_operation = inference_kind === :probabilities ? :predict_proba : :predict
    supported, reason, inference_capabilities =
        _inference_capability(graph, X; operation=inference_operation)
    fit_capabilities = _graph_capabilities(graph, X; phase=:fit)
    skeleton = _reactant_fit_skeleton(graph, X, y, context, weights)
    cpu_graph = skeleton === nothing ?
        Tilia.fit_graph_backend(Tilia.CPUBackend(), graph, X, y,
                                _cpu_context(context), weights) : skeleton
    region_start = 1
    regions = UnitRange{Int}[1:length(graph.nodes)]
    if !supported
        candidate = backend.fallback === :cpu ? _mixed_regions(graph) : nothing
        candidate === nothing && backend.fallback === :cpu &&
            return _fallback_graph(cpu_graph, reason)
        candidate === nothing &&
        throw(Tilia.UnsupportedBackendError(
            "Reactant cannot execute this graph: $reason. Use ReactantBackend(fallback=:cpu) for an explicit host fallback."))
        regions = candidate
        region_start = first(last(regions))
        _apply_mixed_placement!(inference_capabilities, regions, graph)
    end
    numerical_policy = (float_type=context.numerics.float_type,
                        accumulation_type=context.numerics.accumulation_type,
                        deterministic_reductions=context.numerics.deterministic_reductions)
    capability_report = (fit=fit_capabilities, inference=inference_capabilities,
                         predict_proba=inference_capabilities)
    fitted = FittedReactantGraph(cpu_graph, context.cache, backend, numerical_policy,
                                 capability_report, ReentrantLock(), :unknown,
                                 inference_kind, region_start, regions,
                                 supported ? String[] : [reason], UInt64(0), 0,
                                 UInt64(0), UInt64(0), UInt64(0), UInt64(0),
                                 UInt64(0), UInt64(0),
                                 UInt64(0), UInt64(0),
                                 UInt64(0), 0, 0, :host, :host,
                                 collect(Iterators.flatten(regions)),
                                 collect(1:length(graph.nodes)), Int[], NaN)
    try
        _with_reactant_backend(backend.device) do
            fitted.resolved_device = Symbol(lowercase(
                Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
            _accelerate_standardize_fit!(fitted, X, y, context, weights)
            _accelerate_ridge_fit!(fitted, X, y, context, weights)
            _accelerate_logistic_fit!(fitted, X, y, context, weights)
            cpu_graph = fitted.cpu_graph
            region_X = _cpu_prefix_output(cpu_graph, X, region_start)
            _, compile_metrics = _compile_for!(fitted, region_X)
            fitted.compilation_nanoseconds += compile_metrics.compilation_nanoseconds
            fitted.host_conversion_nanoseconds += compile_metrics.host_conversion_nanoseconds
            fitted.transferred_bytes += compile_metrics.estimated_bytes
            if region_start == 1 && inference_kind === :probabilities &&
               all(node -> node.model isa Union{Tilia.Standardize,Tilia.LogisticRegression},
                   graph.nodes)
                objective, objective_metrics = _compile_objective(cpu_graph, X, y, weights)
                fitted.accelerated_objective_value = objective
                fitted.compilation_nanoseconds += objective_metrics.compilation_nanoseconds
                fitted.host_conversion_nanoseconds += objective_metrics.host_conversion_nanoseconds
                fitted.fit_device_execution_nanoseconds += objective_metrics.device_execution_nanoseconds
                fitted.fit_materialization_nanoseconds +=
                    objective_metrics.synchronization_and_materialization_nanoseconds
                fitted.transferred_bytes += objective_metrics.estimated_bytes
            end
        end
    catch error
        error isa Tilia.NumericalFailureError && rethrow()
        backend.fallback === :cpu && return _fallback_graph(
            cpu_graph, "Reactant compilation failed: $(sprint(showerror, error))")
        throw(Tilia.UnsupportedBackendError(
            "Reactant compilation failed for the supported graph: $(sprint(showerror, error)). Use ReactantBackend(fallback=:cpu) for explicit fallback."))
    end
    fitted
end

function _reactant_fit_skeleton(graph, X, y, context, weights)
    X isa AbstractMatrix && Base.nonmissingtype(eltype(X)) in (Float32, Float64) ||
        return nothing
    Missing <: eltype(X) && return nothing
    y isa AbstractVector || return nothing
    models = [node.model for node in graph.nodes]
    has_standardize = length(models) == 2 && graph.edges == [(1, 2)] &&
                      models[1] isa Tilia.Standardize
    head = has_standardize ? models[2] : length(models) == 1 ? models[1] : nothing
    supported_logistic = has_standardize && head isa Tilia.LogisticRegression
    supported_ridge = head isa Tilia.RidgeRegression && head.solver === :cholesky &&
                      head.lambda > 0 && (has_standardize || length(models) == 1)
    supported_logistic || supported_ridge || return nothing

    cpu_context = _cpu_context(context)
    fitted_nodes = Tilia.AbstractFittedEstimator[]
    transformed = X
    if has_standardize
        S = float(eltype(X))
        standardize_report = Tilia.FitReport(observations=size(X, 1),
            features=size(X, 2), backend=:cpu,
            details=(execution=:pending_reactant_statistics,), context=cpu_context)
        standardize = Tilia.FittedStandardize(models[1], zeros(S, size(X, 2)),
            ones(S, size(X, 2)), standardize_report, Tilia.infer_schema(X))
        push!(fitted_nodes, standardize)
    end

    if supported_ridge
        Tilia._validate_regression_data(transformed, y, weights, "RidgeRegression")
        T = float(promote_type(eltype(transformed), eltype(y),
            weights === nothing ? eltype(y) : eltype(weights)))
        placeholder_report = Tilia.FitReport(observations=size(transformed, 1),
            features=size(transformed, 2), backend=:cpu,
            details=(solver=:pending_reactant_cholesky, regularization=head.lambda,
                     weighted=weights !== nothing, fit_intercept=head.fit_intercept),
            context=cpu_context)
        push!(fitted_nodes, Tilia.FittedLinearRegressor(head,
            zeros(T, size(transformed, 2)), zero(T), placeholder_report,
            Tilia.with_target(Tilia.infer_schema(transformed), y)))
    else
        size(transformed, 1) == length(y) || throw(Tilia.SchemaMismatchError(
        "LogisticRegression target has length $(length(y)); expected $(size(transformed, 1)) observations."))
        size(transformed, 1) > 0 && size(transformed, 2) > 0 ||
            throw(Tilia.UnsupportedDataError(
            "LogisticRegression requires at least one observation and feature."))
        classes = Tilia._classification_classes(y)
        T = float(promote_type(eltype(transformed),
                               weights === nothing ? eltype(transformed) : eltype(weights)))
        observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
        length(observation_weights) == length(y) || throw(Tilia.SchemaMismatchError(
        "LogisticRegression weights have length $(length(observation_weights)); expected $(length(y))."))
        all(weight -> isfinite(weight) && weight >= 0, observation_weights) &&
            sum(observation_weights) > 0 || throw(Tilia.UnsupportedDataError(
            "LogisticRegression weights must be finite, nonnegative, and have a positive sum."))
        class_columns = length(classes) == 2 ? 1 : length(classes)
        placeholder_details = (solver=:pending_reactant_newton,
            convergence=falses(class_columns), iterations=zeros(Int, class_columns),
            objective_history=[T[] for _ in 1:class_columns],
            gradient_norms=fill(T(Inf), class_columns), regularization=head.lambda,
            class_order=copy(classes), strategy=:one_vs_rest)
        placeholder_report = Tilia.FitReport(observations=size(transformed, 1),
            features=size(transformed, 2), backend=:cpu, details=placeholder_details,
            context=cpu_context)
        push!(fitted_nodes, Tilia.FittedLogisticRegression(head,
            zeros(T, size(transformed, 2), class_columns), zeros(T, class_columns),
            classes, placeholder_report,
            Tilia.with_class_target(Tilia.infer_schema(transformed), classes)))
    end
    graph_report = Tilia.FitReport(observations=size(X, 1), features=size(X, 2),
        backend=:cpu, details=(nodes=length(models), execution=:reactant_fit_skeleton,
                              weighted=weights !== nothing), context=cpu_context)
    Tilia.FittedGraph(graph, fitted_nodes, graph_report)
end

function _accelerate_logistic_fit!(fitted, X, y, context, weights)
    graph = fitted.cpu_graph.graph
    graph.edges == [(index, index + 1) for index in 1:length(graph.nodes)-1] || return fitted
    model = last(graph.nodes).model
    model isa Tilia.LogisticRegression || return fitted
    head_X = _linear_head_input(fitted.cpu_graph, X)
    Base.nonmissingtype(eltype(head_X)) in (Float32, Float64) || return fitted
    previous = last(fitted.cpu_graph.fitted_nodes)
    P = eltype(previous.coefficients)
    T = fitted.numerical_policy.accumulation_type
    design_features = Matrix{T}(head_X)
    design = model.fit_intercept ?
             hcat(design_features, ones(T, size(design_features, 1))) : design_features
    observation_weights = weights === nothing ? ones(T, length(y)) : Vector{T}(weights)
    penalty_mask = ones(T, size(design, 2))
    model.fit_intercept && (penalty_mask[end] = zero(T))
    penalty_matrix = Matrix(Diagonal(penalty_mask))
    lambda = T[model.lambda]
    tolerance = T[Tilia.effective_tolerance(context, model.tolerance)]
    step_scales = T[T(2)^(-power) for power in 0:20]
    max_iterations = Tilia.effective_max_iterations(context, model.max_iterations)
    trained_classes = length(previous.classes) == 2 ? previous.classes[end:end] : previous.classes
    coefficients = Matrix{P}(undef, size(head_X, 2), length(trained_classes))
    intercepts = Vector{P}(undef, length(trained_classes))
    traces = Vector{Vector{T}}(undef, length(trained_classes))
    iterations = Vector{Int}(undef, length(trained_classes))
    converged = BitVector(undef, length(trained_classes))
    gradient_norms = Vector{T}(undef, length(trained_classes))

    for (column, class) in enumerate(trained_classes)
        target = T.(y .== class)
        entry, compile_metrics = _compile_logistic_newton!(
            fitted, design, target, observation_weights, lambda,
            penalty_mask, penalty_matrix, tolerance, step_scales, max_iterations)
        conversion_started = time_ns()
        host_arrays = (design, target, observation_weights, lambda,
                       penalty_mask, penalty_matrix, tolerance, step_scales)
        device_arrays = map(Reactant.to_rarray, host_arrays)
        fitted.host_conversion_nanoseconds += UInt64(time_ns() - conversion_started) +
            compile_metrics.host_conversion_nanoseconds
        started = time_ns()
        result = entry.compiled(device_arrays...)
        fitted.fit_optimizer_device_execution_nanoseconds += UInt64(time_ns() - started)
        started = time_ns()
        parameters = vec(Array(result[1]))
        iteration_count = Int(round(Reactant.to_number(result[5])))
        factorization_valid = Bool(Reactant.to_number(result[7]))
        factorization_valid || throw(Tilia.NumericalFailureError(
            "Reactant logistic Newton system was singular; increase lambda or remove collinear features."))
        traces[column] = vec(Array(result[2]))[1:iteration_count]
        gradient_norms[column] = T(Reactant.to_number(result[4]))
        iterations[column] = iteration_count
        converged[column] = Bool(Reactant.to_number(result[6]))
        fitted.fit_optimizer_materialization_nanoseconds += UInt64(time_ns() - started)
        coefficients[:, column] = P.(view(parameters, 1:size(head_X, 2)))
        intercepts[column] = model.fit_intercept ? P(parameters[end]) : zero(P)
        fitted.compilation_nanoseconds += compile_metrics.compilation_nanoseconds
        fitted.transferred_bytes += compile_metrics.estimated_bytes +
            sum(Base.summarysize, host_arrays) + Base.summarysize(parameters) +
            Base.summarysize(traces[column])
    end
    warnings = all(converged) ? String[] :
        ["One or more class objectives reached max_iterations before convergence."]
    details = (solver=:reactant_newton, convergence=converged, iterations=iterations,
               objective_history=traces, gradient_norms=gradient_norms,
               regularization=model.lambda, class_order=copy(previous.classes),
               strategy=:one_vs_rest, optimizer_backend=:reactant,
               objective_history_kind=:complete, accumulation_type=T)
    fit_report = Tilia.FitReport(status=all(converged) ? :success : :max_iterations,
        observations=size(head_X, 1), features=size(head_X, 2), backend=:reactant,
        warnings=warnings, details=details, context=context)
    fitted_nodes = copy(fitted.cpu_graph.fitted_nodes)
    fitted_nodes[end] = Tilia.FittedLogisticRegression(
        model, coefficients, intercepts, previous.classes, fit_report, previous.schema)
    fitted.cpu_graph = Tilia.FittedGraph(graph, fitted_nodes, fitted.cpu_graph.report)
    final_id = length(graph.nodes)
    final_id in fitted.accelerator_fit_nodes || push!(fitted.accelerator_fit_nodes, final_id)
    filter!(!=(final_id), fitted.host_fit_nodes)
    fitted
end

function _linear_head_input(cpu_graph, X)
    value = X
    for index in 1:length(cpu_graph.fitted_nodes)-1
        value = Tilia.transform(cpu_graph.fitted_nodes[index], value)
    end
    value
end

function _accelerate_ridge_fit!(fitted, X, y, context, weights)
    graph = fitted.cpu_graph.graph
    graph.edges == [(index, index + 1) for index in 1:length(graph.nodes)-1] || return fitted
    model = last(graph.nodes).model
    model isa Tilia.RidgeRegression || return fitted
    model.solver === :cholesky && model.lambda > 0 || return fitted
    y isa AbstractVector && eltype(y) <: Number || return fitted
    head_X = _linear_head_input(fitted.cpu_graph, X)
    Base.nonmissingtype(eltype(head_X)) in (Float32, Float64) || return fitted

    A = fitted.numerical_policy.accumulation_type
    host_X = Matrix{A}(head_X)
    host_y = Vector{A}(y)
    host_weights = weights === nothing ? ones(A, length(y)) : Vector{A}(weights)
    host_lambda = A[model.lambda]
    host_penalty = Matrix{A}(I, size(host_X, 2), size(host_X, 2))
    entry, compile_metrics = _compile_weighted_ridge_fit!(
        fitted, host_X, model.fit_intercept)
    conversion_started = time_ns()
    host_arrays = (host_X, host_y, host_weights, host_lambda, host_penalty)
    device_arrays = map(Reactant.to_rarray, host_arrays)
    fitted.host_conversion_nanoseconds += UInt64(time_ns() - conversion_started) +
        compile_metrics.host_conversion_nanoseconds
    started = time_ns()
    device_result = entry.compiled(device_arrays...)
    fitted.fit_statistics_device_execution_nanoseconds += UInt64(time_ns() - started)
    started = time_ns()
    coefficients = vec(Array(device_result[1]))
    intercept = only(Array(device_result[2]))
    residual_norm = only(Array(device_result[3]))
    factorization_valid = Bool(Reactant.to_number(device_result[4]))
    fitted.fit_statistics_materialization_nanoseconds += UInt64(time_ns() - started)
    factorization_valid || throw(Tilia.NumericalFailureError(
        "Reactant ridge fit produced a non-positive-definite regularized Gram matrix."))
    result_bytes = Base.summarysize(coefficients) + Base.summarysize(intercept) +
                   Base.summarysize(residual_norm)
    fitted.compilation_nanoseconds += compile_metrics.compilation_nanoseconds
    fitted.transferred_bytes += compile_metrics.estimated_bytes +
        sum(Base.summarysize, host_arrays) + result_bytes
    details = (solver=:cholesky, numerical_rank=size(host_X, 2),
               residual_norm=residual_norm, regularization=model.lambda,
               weighted=weights !== nothing, fit_intercept=model.fit_intercept,
               sufficient_statistics_backend=:reactant,
               solver_backend=:reactant,
               accumulation_type=A,
               deterministic_reductions=fitted.numerical_policy.deterministic_reductions)
    fit_report = Tilia.FitReport(observations=size(host_X, 1),
        features=size(host_X, 2), backend=:reactant, details=details, context=context)
    previous = last(fitted.cpu_graph.fitted_nodes)
    fitted_nodes = copy(fitted.cpu_graph.fitted_nodes)
    fitted_nodes[end] = Tilia.FittedLinearRegressor(model, coefficients, intercept,
                                                     fit_report, previous.schema)
    fitted.cpu_graph = Tilia.FittedGraph(graph, fitted_nodes, fitted.cpu_graph.report)
    final_id = length(graph.nodes)
    final_id in fitted.accelerator_fit_nodes || push!(fitted.accelerator_fit_nodes, final_id)
    filter!(!=(final_id), fitted.host_fit_nodes)
    fitted
end

function _accelerate_standardize_fit!(fitted, X, y, context, weights)
    graph = fitted.cpu_graph.graph
    graph.edges == [(index, index + 1) for index in 1:length(graph.nodes)-1] || return fitted
    first(graph.nodes).model isa Tilia.Standardize || return fitted
    X isa AbstractMatrix && Base.nonmissingtype(eltype(X)) in (Float32, Float64) || return fitted
    Missing <: eltype(X) && return fitted

    entry, compile_metrics = _compile_standardize_statistics!(fitted, X)
    A = fitted.numerical_policy.accumulation_type
    conversion_started = time_ns()
    host_input = Matrix{A}(X)
    device_input = Reactant.to_rarray(host_input)
    fitted.host_conversion_nanoseconds += UInt64(time_ns() - conversion_started) +
        compile_metrics.host_conversion_nanoseconds
    started = time_ns()
    device_means, device_m2 = entry.compiled(device_input)
    fitted.fit_statistics_device_execution_nanoseconds += UInt64(time_ns() - started)
    started = time_ns()
    running_mean = vec(Array(device_means))
    m2_accumulated = vec(Array(device_m2))
    fitted.fit_statistics_materialization_nanoseconds += UInt64(time_ns() - started)
    fitted.compilation_nanoseconds += compile_metrics.compilation_nanoseconds
    fitted.transferred_bytes += compile_metrics.estimated_bytes + Base.summarysize(host_input) +
        Base.summarysize(running_mean) + Base.summarysize(m2_accumulated)

    previous = first(fitted.cpu_graph.fitted_nodes)
    T = eltype(previous.means)
    typed_mean = T.(running_mean)
    typed_m2 = T.(m2_accumulated)
    means = previous.model.center ? typed_mean : zeros(T, size(X, 2))
    raw_scales = previous.model.scale ? T.(sqrt.(A.(typed_m2) ./ A(size(X, 1)))) :
                 ones(T, size(X, 2))
    scales = map(value -> iszero(value) ? one(value) : value, raw_scales)
    details = merge(previous.report.details, (
        running_mean=typed_mean, m2=typed_m2,
        zero_variance=count(iszero, raw_scales),
        sufficient_statistics_backend=:reactant,
        accumulation_type=A,
        deterministic_reductions=fitted.numerical_policy.deterministic_reductions))
    statistics_report = Tilia.FitReport(
        observations=size(X, 1), features=size(X, 2), backend=:reactant,
        details=details, context=context)
    fitted_nodes = copy(fitted.cpu_graph.fitted_nodes)
    fitted_nodes[1] = Tilia.FittedStandardize(previous.model, means, scales,
                                               statistics_report, previous.schema)
    value = Tilia.transform(fitted_nodes[1], X)
    cpu_context = _cpu_context(context)
    for index in 2:length(fitted_nodes)
        semantic = graph.nodes[index]
        node_context = Tilia.derive_context(cpu_context, :graph_node, index)
        if semantic isa Tilia.TransformNode
            fitted_nodes[index] = Tilia.fit(semantic.model, value; context=node_context)
            value = Tilia.transform(fitted_nodes[index], value)
        else
            if semantic.model isa Tilia.LogisticRegression ||
               (semantic.model isa Tilia.RidgeRegression &&
                semantic.model.solver === :cholesky && semantic.model.lambda > 0)
                continue
            end
            fitted_nodes[index] = Tilia.fit(semantic.model, value, y;
                                            weights=weights, context=node_context)
        end
    end
    fitted.cpu_graph = Tilia.FittedGraph(graph, fitted_nodes, fitted.cpu_graph.report)
    fitted.accelerator_fit_nodes = [1]
    filter!(!=(1), fitted.host_fit_nodes)
    fitted
end

function _validate_output_location(output::Symbol)
    output in (:host, :device) || throw(ArgumentError(
        "output must be :host or :device; received :$output."))
    output
end

function Tilia.predict_proba(fitted::FittedReactantGraph, X::AbstractMatrix;
                             output::Symbol=:host)
    _validate_output_location(output)
    fitted.inference_kind === :probabilities || throw(Tilia.UnsupportedDataError(
        "predict_proba is unavailable for a Reactant regression graph."))
    size(X, 1) > 0 || throw(Tilia.UnsupportedDataError(
        "Reactant prediction requires a nonempty observation batch."))
    lock(fitted.execution_lock) do
        _with_reactant_backend(fitted.backend.device) do
            _predict_proba_locked(fitted, _prepare_prediction_input_locked(fitted, X); output)
        end
    end
end


function _execute_transform_region_locked(fitted, X, region)
    parameters = _transform_region_parameters(fitted.cpu_graph, region)
    size(X, 2) == parameters.input_features || throw(Tilia.SchemaMismatchError(
        "Reactant transform region expects $(parameters.input_features) features; received $(size(X, 2))."))
    entry, metrics = _compile_transform_region!(fitted, X, region)
    fitted.compilation_nanoseconds += metrics.compilation_nanoseconds
    fitted.host_conversion_nanoseconds += metrics.host_conversion_nanoseconds
    host_arrays = (Matrix(X), _transform_parameter_arrays(parameters)...)
    conversion_started = time_ns()
    device_arrays = map(Reactant.to_rarray, host_arrays)
    fitted.host_conversion_nanoseconds += UInt64(time_ns() - conversion_started)
    started = time_ns()
    device_result = entry.compiled(device_arrays...)
    fitted.last_execution_nanoseconds = UInt64(time_ns() - started)
    started = time_ns()
    result = Array(device_result)
    fitted.last_materialization_nanoseconds = UInt64(time_ns() - started)
    fitted.last_result_estimated_bytes = Base.summarysize(result)
    fitted.transferred_bytes += metrics.estimated_bytes +
        sum(Base.summarysize, host_arrays) + fitted.last_result_estimated_bytes
    result
end

function _prepare_prediction_input_locked(fitted, X)
    fitted.region_start == 1 && return X
    graph = fitted.cpu_graph.graph
    if graph.edges != [(index, index + 1) for index in 1:length(graph.nodes)-1]
        predecessors = Tilia.graph_predecessors(graph)
        values = Vector{Any}(undef, length(graph.nodes))
        regions = Dict(first(region) => region for region in fitted.regions
                       if last(region) < fitted.region_start)
        index = 1
        while index < fitted.region_start
            input = Tilia._graph_input(values, predecessors[index], X)
            if haskey(regions, index)
                region = regions[index]
                value = _execute_transform_region_locked(fitted, input, region)
                values[last(region)] = value
                index = last(region) + 1
            else
                values[index] = Tilia.transform(fitted.cpu_graph.fitted_nodes[index], input)
                index += 1
            end
        end
        return Tilia._graph_input(values, predecessors[fitted.region_start], X)
    end
    value = X
    index = 1
    for region in fitted.regions
        first(region) >= fitted.region_start && break
        while index < first(region)
            value = Tilia.transform(fitted.cpu_graph.fitted_nodes[index], value)
            index += 1
        end
        value = _execute_transform_region_locked(fitted, value, region)
        index = last(region) + 1
    end
    while index < fitted.region_start
        value = Tilia.transform(fitted.cpu_graph.fitted_nodes[index], value)
        index += 1
    end
    value
end

function _predict_proba_locked(fitted::FittedReactantGraph, X::AbstractMatrix;
                               output::Symbol=:host)
    _execute_reactant_locked(fitted, X; operation=:output, output)
end

function _execute_reactant_locked(fitted::FittedReactantGraph, X::AbstractMatrix;
                                  operation::Symbol, output::Symbol=:host)
    parameters = _linear_region_parameters(fitted.cpu_graph;
                                           region_start=fitted.region_start)
    size(X, 2) == parameters.input_features || throw(Tilia.SchemaMismatchError(
        "Reactant graph was fitted with $(parameters.input_features) features; received $(size(X, 2))."))
    entry, compile_metrics = _compile_for!(fitted, X; operation)
    fitted.compilation_nanoseconds += compile_metrics.compilation_nanoseconds
    fitted.host_conversion_nanoseconds += compile_metrics.host_conversion_nanoseconds
    conversion_started = time_ns()
    device_arrays = _device_arrays(parameters, X)
    fitted.host_conversion_nanoseconds += UInt64(time_ns() - conversion_started)
    started = time_ns()
    device_result = entry.compiled(device_arrays...)
    fitted.last_execution_nanoseconds = UInt64(time_ns() - started)
    fitted.last_input_location = _is_device_array(X) ? :device : :host
    fitted.last_output_location = output
    if output === :device
        fitted.last_materialization_nanoseconds = UInt64(0)
        fitted.last_result_estimated_bytes = 0
        result = device_result
    else
        started = time_ns()
        result = Array(device_result)
        fitted.last_materialization_nanoseconds = UInt64(time_ns() - started)
        fitted.last_result_estimated_bytes = Base.summarysize(result)
    end
    fitted.transferred_bytes += compile_metrics.estimated_bytes +
        _input_transfer_estimate(parameters, X) + fitted.last_result_estimated_bytes
    result
end

function Tilia.predict(fitted::FittedReactantGraph, X::AbstractMatrix;
                       output::Symbol=:host)
    _validate_output_location(output)
    if fitted.inference_kind === :probabilities
        output === :host || throw(Tilia.UnsupportedDataError(
            "device output is unavailable for classification labels; use predict_proba(...; output=:device) for device-resident probabilities."))
        size(X, 1) > 0 || throw(Tilia.UnsupportedDataError(
            "Reactant prediction requires a nonempty observation batch."))
        indices = lock(fitted.execution_lock) do
            _with_reactant_backend(fitted.backend.device) do
                region_X = _prepare_prediction_input_locked(fitted, X)
                _execute_reactant_locked(fitted, region_X; operation=:class_indices)
            end
        end
        classes = last(fitted.cpu_graph.fitted_nodes).classes
        return classes[Int.(indices)]
    end
    size(X, 1) > 0 || throw(Tilia.UnsupportedDataError(
        "Reactant prediction requires a nonempty observation batch."))
    lock(fitted.execution_lock) do
        _with_reactant_backend(fitted.backend.device) do
            _predict_proba_locked(fitted, _prepare_prediction_input_locked(fitted, X); output)
        end
    end
end
