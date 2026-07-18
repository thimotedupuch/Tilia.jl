function _compile_objective(cpu_graph, X, y, observation_weights=nothing)
    logistic = last(cpu_graph.fitted_nodes)
    T = eltype(logistic.coefficients)
    standardize = if length(cpu_graph.fitted_nodes) == 1
        (means=zeros(T, size(logistic.coefficients, 1)),
         scales=ones(T, size(logistic.coefficients, 1)))
    else
        cpu_graph.fitted_nodes[1]
    end
    trained_classes = length(logistic.classes) == 2 ? logistic.classes[end:end] : logistic.classes
    target = hcat((T.(y .== class) for class in trained_classes)...)
    weights = observation_weights === nothing ? ones(T, length(y), 1) :
              reshape(T.(observation_weights), :, 1)
    lambda = T[logistic.model.lambda]
    conversion_started = time_ns()
    host_arrays = (Matrix{T}(X), target, weights, standardize.means,
                   standardize.scales, logistic.coefficients, logistic.intercept, lambda)
    transferred = sum(Base.summarysize, host_arrays)
    device_arrays = map(Reactant.to_rarray, host_arrays)
    conversion_elapsed = UInt64(time_ns() - conversion_started)
    started = time_ns()
    compiled = Reactant.compile(_reactant_logistic_objective, device_arrays)
    compilation_elapsed = UInt64(time_ns() - started)
    started = time_ns()
    result = compiled(device_arrays...)
    execution_elapsed = UInt64(time_ns() - started)
    started = time_ns()
    objective = Reactant.to_number(result)
    materialization_elapsed = UInt64(time_ns() - started)
    objective, (host_conversion_nanoseconds=conversion_elapsed,
                compilation_nanoseconds=compilation_elapsed,
                device_execution_nanoseconds=execution_elapsed,
                synchronization_and_materialization_nanoseconds=materialization_elapsed,
                estimated_bytes=transferred)
end
