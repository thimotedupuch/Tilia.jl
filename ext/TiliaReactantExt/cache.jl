function _cached_entry(cache, key)
    haskey(cache.entries, key) || return nothing
    filter!(!=(key), cache.order)
    push!(cache.order, key)
    cache.entries[key]
end

function _store_compilation!(cache, key, entry)
    while length(cache.entries) >= cache.max_entries && !isempty(cache.order)
        evicted = popfirst!(cache.order)
        delete!(cache.entries, evicted)
        cache.evictions += 1
    end
    cache.entries[key] = entry
    push!(cache.order, key)
    cache.compilations += 1
    entry
end

function _compile_for!(fitted, X::AbstractMatrix; operation::Symbol=:output)
    parameters = _linear_region_parameters(fitted.cpu_graph;
                                           region_start=fitted.region_start)
    parameter_arrays = _parameter_arrays(parameters)
    parameter_metadata = map(array -> (eltype(array), size(array)), parameter_arrays)
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    key = UInt64(hash((:tilia_reactant_linear_region_v4, parameters.kind,
                       fitted.inference_kind, operation,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       eltype(X), size(X), parameter_metadata)))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached,
                   (host_conversion_nanoseconds=UInt64(0),
                    compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        conversion_started = time_ns()
        transferred = _input_transfer_estimate(parameters, X)
        device_arrays = _device_arrays(parameters, X)
        conversion_elapsed = UInt64(time_ns() - conversion_started)
        started = time_ns()
        kernel = if parameters.kind === :imputed
            operation === :class_indices ? _reactant_imputed_class_indices :
            fitted.inference_kind === :probabilities ?
            _reactant_imputed_probabilities : _reactant_imputed_regression
        elseif parameters.kind === :clipped
            operation === :class_indices ? _reactant_clipped_class_indices :
            fitted.inference_kind === :probabilities ?
            _reactant_clipped_probabilities : _reactant_clipped_regression
        else
            operation === :class_indices ? _reactant_class_indices :
            fitted.inference_kind === :probabilities ?
            _reactant_probabilities : _reactant_regression
        end
        compiled = Reactant.compile(kernel, device_arrays)
        elapsed = UInt64(time_ns() - started)
        # Cache only the executable. Parameters belong to a particular fitted
        # graph and must never be retained by a shared compilation cache.
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        entry, (host_conversion_nanoseconds=conversion_elapsed,
                compilation_nanoseconds=elapsed, estimated_bytes=transferred)
    end
end


function _compile_transform_region!(fitted, X::AbstractMatrix, region::UnitRange{Int})
    parameters = _transform_region_parameters(fitted.cpu_graph, region)
    parameter_arrays = _transform_parameter_arrays(parameters)
    metadata = map(array -> (eltype(array), size(array)), parameter_arrays)
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    key = UInt64(hash((:tilia_reactant_transform_region_v2, parameters.kind, region,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       eltype(X), size(X), metadata)))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached,
                   (host_conversion_nanoseconds=UInt64(0),
                    compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        conversion_started = time_ns()
        host_arrays = (Matrix(X), parameter_arrays...)
        estimated = sum(Base.summarysize, host_arrays)
        device_arrays = map(Reactant.to_rarray, host_arrays)
        conversion_elapsed = UInt64(time_ns() - conversion_started)
        started = time_ns()
        kernel = parameters.kind === :clipped ?
                 _reactant_clipped_transform_region : _reactant_transform_region
        compiled = Reactant.compile(kernel, device_arrays)
        compilation_elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        entry, (host_conversion_nanoseconds=conversion_elapsed,
                compilation_nanoseconds=compilation_elapsed,
                estimated_bytes=estimated)
    end
end

function _compile_standardize_statistics!(fitted, X::AbstractMatrix)
    A = fitted.numerical_policy.accumulation_type
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    key = UInt64(hash((:tilia_reactant_standardize_statistics_v1,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       A, size(X))))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached, (host_conversion_nanoseconds=UInt64(0),
                            compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        conversion_started = time_ns()
        host_input = Matrix{A}(X)
        device_input = Reactant.to_rarray(host_input)
        conversion_elapsed = UInt64(time_ns() - conversion_started)
        started = time_ns()
        compiled = Reactant.compile(_reactant_standardize_statistics, (device_input,))
        compilation_elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        entry, (host_conversion_nanoseconds=conversion_elapsed,
                compilation_nanoseconds=compilation_elapsed,
                estimated_bytes=Base.summarysize(host_input))
    end
end

function _compile_weighted_regression_statistics!(fitted, X::AbstractMatrix)
    A = fitted.numerical_policy.accumulation_type
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    key = UInt64(hash((:tilia_reactant_weighted_regression_statistics_v1,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       A, size(X))))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached, (host_conversion_nanoseconds=UInt64(0),
                            compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        sample_X = Reactant.to_rarray(Matrix{A}(X))
        sample_target = Reactant.to_rarray(zeros(A, size(X, 1)))
        sample_weights = Reactant.to_rarray(ones(A, size(X, 1)))
        estimated = Base.summarysize(X) + Base.summarysize(sample_target) +
                    Base.summarysize(sample_weights)
        started = time_ns()
        compiled = Reactant.compile(_reactant_weighted_regression_statistics,
                                    (sample_X, sample_target, sample_weights))
        compilation_elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        entry, (host_conversion_nanoseconds=UInt64(0),
                compilation_nanoseconds=compilation_elapsed,
                estimated_bytes=estimated)
    end
end

function _compile_weighted_ridge_fit!(fitted, X::AbstractMatrix, fit_intercept::Bool)
    A = fitted.numerical_policy.accumulation_type
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    key = UInt64(hash((:tilia_reactant_weighted_ridge_fit_v1, fit_intercept,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       A, size(X))))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached, (host_conversion_nanoseconds=UInt64(0),
                            compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        sample_X = Reactant.to_rarray(Matrix{A}(X))
        sample_target = Reactant.to_rarray(zeros(A, size(X, 1)))
        sample_weights = Reactant.to_rarray(ones(A, size(X, 1)))
        sample_lambda = Reactant.to_rarray(A[one(A)])
        sample_penalty = Reactant.to_rarray(Matrix{A}(I, size(X, 2), size(X, 2)))
        implementation = fit_intercept ? _reactant_weighted_ridge_fit_intercept :
                                         _reactant_weighted_ridge_fit_no_intercept
        started = time_ns()
        compiled = Reactant.compile(implementation,
            (sample_X, sample_target, sample_weights, sample_lambda, sample_penalty))
        compilation_elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        estimated = sum(Base.summarysize,
                        (sample_X, sample_target, sample_weights, sample_lambda,
                         sample_penalty))
        entry, (host_conversion_nanoseconds=UInt64(0),
                compilation_nanoseconds=compilation_elapsed,
                estimated_bytes=estimated)
    end
end

function _compile_logistic_newton!(fitted, design, target, weights, lambda,
                                   penalty_mask, penalty_matrix, tolerance,
                                   step_scales, max_iterations::Int)
    platform = Symbol(lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    metadata = map(array -> (eltype(array), size(array)),
                   (design, target, weights, lambda, penalty_mask,
                    penalty_matrix, tolerance, step_scales))
    key = UInt64(hash((:tilia_reactant_logistic_newton_v1, max_iterations,
                       fitted.backend.device, platform, fitted.numerical_policy,
                       metadata)))
    lock(fitted.cache.lock) do
        cached = _cached_entry(fitted.cache, key)
        if cached !== nothing
            fitted.cache_hits += 1
            return cached, (host_conversion_nanoseconds=UInt64(0),
                            compilation_nanoseconds=UInt64(0), estimated_bytes=0)
        end
        conversion_started = time_ns()
        host_arrays = (design, target, weights, lambda, penalty_mask,
                       penalty_matrix, tolerance, step_scales)
        device_arrays = map(Reactant.to_rarray, host_arrays)
        conversion_elapsed = UInt64(time_ns() - conversion_started)
        kernel = (args...) -> _reactant_binary_logistic_newton(
            args..., Val(max_iterations))
        started = time_ns()
        compiled = Reactant.compile(kernel, device_arrays)
        compilation_elapsed = UInt64(time_ns() - started)
        entry = (compiled=compiled,)
        _store_compilation!(fitted.cache, key, entry)
        entry, (host_conversion_nanoseconds=conversion_elapsed,
                compilation_nanoseconds=compilation_elapsed,
                estimated_bytes=sum(Base.summarysize, host_arrays))
    end
end
