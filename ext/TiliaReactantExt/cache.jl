function _compile_for!(fitted, X::AbstractMatrix)
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
