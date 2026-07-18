function _arrays(fitted)
    logistic = last(fitted.cpu_graph.fitted_nodes)
    if length(fitted.cpu_graph.fitted_nodes) == 1
        T = eltype(logistic.coefficients)
        features = size(logistic.coefficients, 1)
        return (means=zeros(T, features), scales=ones(T, features)), logistic
    end
    fitted.cpu_graph.fitted_nodes[1], logistic
end

function _linear_region_parameters(cpu_graph; region_start::Int=1)
    fitted_region = cpu_graph.fitted_nodes[region_start:end]
    head = last(fitted_region)
    coefficients = head.coefficients
    T = eltype(coefficients)
    features = size(coefficients, 1)
    input_features = region_start == 1 ? cpu_graph.report.features :
                     Tilia.nfeatures(first(fitted_region).schema)
    linear_edges = [(index, index + 1) for index in 1:length(cpu_graph.graph.nodes)-1]
    region_start == 1 && (cpu_graph.graph.edges == linear_edges &&
     !any(fitted -> fitted isa Union{Tilia.FittedSelect,Tilia.FittedConcatenate},
          cpu_graph.fitted_nodes)) ||
        region_start > 1 || return _branched_linear_parameters(cpu_graph, T, input_features)
    projection = Matrix{T}(I, input_features, input_features)
    offset = zeros(T, input_features)
    clip_parameters = nothing
    imputation_fill = nothing
    for fitted in fitted_region[1:end-1]
        if fitted isa Tilia.FittedImpute
            imputation_fill === nothing || throw(Tilia.UnsupportedBackendError(
                "Reactant linear regions support one imputation operation."))
            imputation_fill = T[fill for fill in fitted.fill_values]
        elseif fitted isa Tilia.FittedStandardize
            scales = T.(fitted.scales)
            inverse_scales = one(T) ./ scales
            projection .*= transpose(inverse_scales)
            offset .= (offset .- T.(fitted.means)) ./ scales
        elseif fitted isa Tilia.FittedMinMaxScale
            lower, upper = T.(fitted.model.feature_range)
            ranges = map(value -> iszero(value) ? one(T) : T(value), fitted.ranges)
            next_multiplier = (upper - lower) ./ ranges
            offset .= lower .+ (offset .- T.(fitted.minima)) .* next_multiplier
            projection .*= transpose(next_multiplier)
            if fitted.model.clip
                clip_parameters === nothing || throw(Tilia.UnsupportedBackendError(
                    "Reactant linear-region lowering supports at most one clipping operation."))
                clip_parameters = (coefficients=projection, offset=offset,
                                   lower=T[lower], upper=T[upper])
                dimension = size(projection, 2)
                projection = Matrix{T}(I, dimension, dimension)
                offset = zeros(T, dimension)
            end
        elseif fitted isa Tilia.FittedDecomposition
            next_projection = T.(fitted.components)
            if fitted.model isa Tilia.PCA && fitted.model.whiten
                scales = sqrt.(T.(fitted.explained_variance))
                safe_scales = map(scale -> iszero(scale) ? one(T) : scale, scales)
                next_projection ./= transpose(safe_scales)
            end
            offset = transpose(next_projection) * (offset .- T.(fitted.mean))
            projection = projection * next_projection
        else
            throw(Tilia.UnsupportedBackendError(
                "Reactant has no affine parameter lowering for $(nameof(typeof(fitted)))."))
        end
    end
    if coefficients isa AbstractVector
        effective_coefficients = projection * coefficients
        intercept = T[head.intercept + dot(offset, coefficients)]
    else
        effective_coefficients = projection * coefficients
        intercept = T.(head.intercept) .+ vec(transpose(offset) * coefficients)
    end
    if clip_parameters === nothing
        imputation_fill === nothing || return (
            kind=:imputed, input_features=input_features, fill_values=imputation_fill,
            coefficients=effective_coefficients, intercept=intercept)
        return (kind=:linear, input_features=input_features,
                coefficients=effective_coefficients, intercept=intercept)
    end
    (kind=:clipped, input_features=input_features,
     preclip_coefficients=clip_parameters.coefficients,
     preclip_offset=clip_parameters.offset, lower=clip_parameters.lower,
     upper=clip_parameters.upper, coefficients=effective_coefficients,
     intercept=intercept)
end

function _branched_linear_parameters(cpu_graph, ::Type{T}, input_features) where {T}
    predecessors = Tilia.graph_predecessors(cpu_graph.graph)
    maps = Vector{Any}(undef, length(cpu_graph.fitted_nodes))
    identity_projection = Matrix{T}(I, input_features, input_features)
    identity_offset = zeros(T, input_features)
    for (index, fitted) in enumerate(cpu_graph.fitted_nodes)
        incoming = predecessors[index]
        if fitted isa Tilia.FittedConcatenate
            projection = reduce(hcat, (maps[id].projection for id in incoming))
            offset = reduce(vcat, (maps[id].offset for id in incoming))
        else
            source = isempty(incoming) ?
                     (projection=identity_projection, offset=identity_offset) :
                     only(maps[incoming])
            projection = copy(source.projection)
            offset = copy(source.offset)
        end

        if fitted isa Tilia.FittedSelect
            projection = projection[:, fitted.indices]
            offset = offset[fitted.indices]
        elseif fitted isa Tilia.FittedStandardize
            scales = T.(fitted.scales)
            inverse_scales = one(T) ./ scales
            projection .*= transpose(inverse_scales)
            offset .= (offset .- T.(fitted.means)) ./ scales
        elseif fitted isa Tilia.FittedMinMaxScale
            fitted.model.clip && throw(Tilia.UnsupportedBackendError(
                "Reactant branched linear regions do not yet support clipping."))
            lower, upper = T.(fitted.model.feature_range)
            ranges = map(value -> iszero(value) ? one(T) : T(value), fitted.ranges)
            multiplier = (upper - lower) ./ ranges
            projection .*= transpose(multiplier)
            offset .= lower .+ (offset .- T.(fitted.minima)) .* multiplier
        elseif fitted isa Tilia.FittedDecomposition
            next_projection = T.(fitted.components)
            if fitted.model isa Tilia.PCA && fitted.model.whiten
                scales = sqrt.(T.(fitted.explained_variance))
                next_projection ./= transpose(map(scale -> iszero(scale) ? one(T) : scale,
                                                   scales))
            end
            offset = transpose(next_projection) * (offset .- T.(fitted.mean))
            projection = projection * next_projection
        elseif fitted isa Tilia.FittedConcatenate
            nothing
        elseif fitted isa Union{Tilia.FittedLinearRegressor,Tilia.FittedLogisticRegression}
            coefficients = fitted.coefficients
            if coefficients isa AbstractVector
                effective_coefficients = projection * coefficients
                intercept = T[fitted.intercept + dot(offset, coefficients)]
            else
                effective_coefficients = projection * coefficients
                intercept = T.(fitted.intercept) .+ vec(transpose(offset) * coefficients)
            end
            return (kind=:linear, input_features=input_features,
                    coefficients=effective_coefficients, intercept=intercept)
        else
            throw(Tilia.UnsupportedBackendError(
                "Reactant has no branched linear lowering for $(nameof(typeof(fitted)))."))
        end
        maps[index] = (projection=projection, offset=offset)
    end
    throw(Tilia.UnsupportedBackendError("Reactant branched region has no prediction head."))
end

_parameter_arrays(parameters) = parameters.kind === :linear ?
    (parameters.coefficients, parameters.intercept) :
    parameters.kind === :imputed ?
    (parameters.fill_values, parameters.coefficients, parameters.intercept) :
    (parameters.preclip_coefficients, parameters.preclip_offset, parameters.lower,
     parameters.upper, parameters.coefficients, parameters.intercept)

function _host_arrays(parameters, X)
    if parameters.kind === :imputed
        T = eltype(parameters.coefficients)
        mask = ismissing.(X)
        values = map(value -> ismissing(value) ? zero(T) : T(value), X)
        return (Matrix{T}(values), mask, _parameter_arrays(parameters)...)
    end
    (Matrix(X), _parameter_arrays(parameters)...)
end

_is_device_array(X) = X isa Reactant.AbstractConcreteArray

function _device_arrays(parameters, X)
    input = _is_device_array(X) ? X : Reactant.to_rarray(first(_host_arrays(parameters, X)))
    if parameters.kind === :imputed
        T = eltype(parameters.coefficients)
        if _is_device_array(X)
            mask = Reactant.to_rarray(falses(size(X)))
            return (input, mask, map(Reactant.to_rarray, _parameter_arrays(parameters))...)
        end
        return map(Reactant.to_rarray, _host_arrays(parameters, X))
    end
    (input, map(Reactant.to_rarray, _parameter_arrays(parameters))...)
end

function _input_transfer_estimate(parameters, X)
    parameter_bytes = sum(Base.summarysize, _parameter_arrays(parameters))
    _is_device_array(X) && return parameter_bytes +
        (parameters.kind === :imputed ? Base.summarysize(falses(size(X))) : 0)
    sum(Base.summarysize, _host_arrays(parameters, X))
end

function _transform_region_parameters(cpu_graph, region::UnitRange{Int})
    first_fitted = cpu_graph.fitted_nodes[first(region)]
    T = if first_fitted isa Tilia.FittedSelect
        Base.nonmissingtype(first_fitted.schema.columns[1].physical_type)
    elseif hasproperty(first_fitted, :means)
        eltype(first_fitted.means)
    elseif hasproperty(first_fitted, :components)
        eltype(first_fitted.components)
    elseif hasproperty(first_fitted, :minima)
        eltype(first_fitted.minima)
    else
        Float64
    end
    input_features = Tilia.nfeatures(first_fitted.schema)
    projection = Matrix{T}(I, input_features, input_features)
    offset = zeros(T, input_features)
    clip_parameters = nothing
    for index in region
        fitted = cpu_graph.fitted_nodes[index]
        if fitted isa Tilia.FittedSelect
            projection = projection[:, fitted.indices]
            offset = offset[fitted.indices]
        elseif fitted isa Tilia.FittedStandardize
            scales = T.(fitted.scales)
            inverse_scales = one(T) ./ scales
            projection .*= transpose(inverse_scales)
            offset .= (offset .- T.(fitted.means)) ./ scales
        elseif fitted isa Tilia.FittedMinMaxScale
            lower, upper = T.(fitted.model.feature_range)
            ranges = map(value -> iszero(value) ? one(T) : T(value), fitted.ranges)
            multiplier = (upper - lower) ./ ranges
            projection .*= transpose(multiplier)
            offset .= lower .+ (offset .- T.(fitted.minima)) .* multiplier
            if fitted.model.clip
                clip_parameters = (coefficients=projection, offset=offset,
                                   lower=T[lower], upper=T[upper])
                dimension = size(projection, 2)
                projection = Matrix{T}(I, dimension, dimension)
                offset = zeros(T, dimension)
            end
        elseif fitted isa Tilia.FittedDecomposition
            next_projection = T.(fitted.components)
            if fitted.model isa Tilia.PCA && fitted.model.whiten
                scales = sqrt.(T.(fitted.explained_variance))
                next_projection ./= transpose(map(scale -> iszero(scale) ? one(T) : scale,
                                                   scales))
            end
            offset = transpose(next_projection) * (offset .- T.(fitted.mean))
            projection = projection * next_projection
        else
            throw(Tilia.UnsupportedBackendError(
                "Reactant cannot lower transform-only region node $(index)."))
        end
    end
    clip_parameters === nothing && return (
        kind=:affine, input_features=input_features,
        coefficients=projection, offset=offset)
    (kind=:clipped, input_features=input_features,
     preclip_coefficients=clip_parameters.coefficients,
     preclip_offset=clip_parameters.offset,
     lower=clip_parameters.lower, upper=clip_parameters.upper,
     coefficients=projection, offset=offset)
end

_transform_parameter_arrays(parameters) = parameters.kind === :affine ?
    (parameters.coefficients, parameters.offset) :
    (parameters.preclip_coefficients, parameters.preclip_offset,
     parameters.lower, parameters.upper, parameters.coefficients, parameters.offset)
