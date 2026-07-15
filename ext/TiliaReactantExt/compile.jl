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
