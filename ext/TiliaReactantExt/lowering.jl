const _REACTANT_PRIMITIVE_SUPPORT = Dict(
    (:inference, :normalize) => true,
    (:inference, :center) => true,
    (:inference, :clamp) => true,
    (:inference, :gather) => true,
    (:inference, :concatenate) => true,
    (:inference, :missing_mask) => true,
    (:inference, :select_fill) => true,
    (:inference, :transfer_host_to_device) => true,
    (:inference, :transfer_device_to_host) => true,
    (:inference, :affine) => true,
    (:inference, :matmul) => true,
    (:inference, :add) => true,
    (:inference, :sigmoid) => true,
)

const _REACTANT_LOWERING_REGISTRY = (
    transforms=(Tilia.Standardize, Tilia.MinMaxScale, Tilia.PCA, Tilia.TruncatedSVD,
                Tilia.Select, Tilia.Concatenate, Tilia.Impute),
    probabilistic_heads=(Tilia.LogisticRegression,),
    regression_heads=(Tilia.LinearRegression, Tilia.RidgeRegression),
)

_registered_transform(model) = any(T -> model isa T, _REACTANT_LOWERING_REGISTRY.transforms)
_probabilistic_head(model) = any(T -> model isa T, _REACTANT_LOWERING_REGISTRY.probabilistic_heads)
_regression_head(model) = any(T -> model isa T, _REACTANT_LOWERING_REGISTRY.regression_heads)

const _REACTANT_BACKEND_LOCK = ReentrantLock()

function _with_reactant_backend(f, requested::Symbol)
    lock(_REACTANT_BACKEND_LOCK) do
        previous = Reactant.XLA.default_backend()
        try
            requested !== :auto && Reactant.set_default_backend(string(requested))
            f()
        finally
            Reactant.set_default_backend(previous)
        end
    end
end

function _primitive_capabilities(numerical::Tilia.NumericalExecutionGraph)
    [(node_id=primitive.semantic_node_id,
      primitive_id=primitive.id,
      operation=primitive.operation,
      phase=numerical.phase,
      supported=get(_REACTANT_PRIMITIVE_SUPPORT,
                    (numerical.phase, primitive.operation), false),
      reason=get(_REACTANT_PRIMITIVE_SUPPORT,
                 (numerical.phase, primitive.operation), false) ? "" :
          "Reactant does not lower $(primitive.operation) during $(numerical.phase)")
     for primitive in numerical.primitives]
end

function _graph_capabilities(graph, X; phase::Symbol=:inference,
                             operation::Symbol=:predict_proba)
    numerical = Tilia.lower_graph(graph, X; phase=phase, operation=operation,
                                  device=:reactant)
    (graph=numerical, primitives=_primitive_capabilities(numerical))
end

function _inference_capability(graph, X; operation::Symbol=:predict_proba)
    X isa AbstractMatrix || return false,
        "Reactant $(operation) requires a dense matrix input", nothing
    models = [node.model for node in graph.nodes
              if node isa Union{Tilia.TransformNode,Tilia.PredictorNode}]
    input_type = Base.nonmissingtype(eltype(X))
    input_type in (Float32, Float64) || return false,
        "Reactant $(operation) supports Float32 and Float64 inputs; received $(eltype(X))", nothing
    allows_missing = !isempty(models) && first(models) isa Tilia.Impute
    Missing <: eltype(X) && !allows_missing && return false,
        "Reactant $(operation) requires Impute as the first operation for missing input", nothing

    capabilities = _graph_capabilities(graph, X; phase=:inference, operation)
    unsupported = filter(record -> !record.supported, capabilities.primitives)
    isempty(unsupported) || return false, first(unsupported).reason, capabilities

    isempty(models) && return false,
        "Reactant $(operation) requires a lowered predictor node", capabilities
    transforms = models[1:end-1]
    head = last(models)
    composable = all(_registered_transform, transforms) &&
                 (_probabilistic_head(head) || _regression_head(head)) &&
                 count(model -> model isa Tilia.MinMaxScale && model.clip,
                       transforms) <= 1 &&
                 !(any(model -> model isa Tilia.Impute, transforms) &&
                   any(model -> model isa Tilia.MinMaxScale && model.clip, transforms)) &&
                 (length(graph.edges) == length(graph.nodes) - 1 ||
                  !any(model -> model isa Union{Tilia.MinMaxScale,Tilia.Impute} &&
                       (model isa Tilia.Impute || model.clip), transforms))
    composable || return false,
        "Reactant cannot compose the supported $(operation) primitives for this graph topology", capabilities
    true, "", capabilities
end

function _cpu_context(context)
    Tilia.FitContext(backend=Tilia.CPUBackend(), rng=context.rng,
        numerics=context.numerics, deterministic=context.deterministic,
        cache=context.cache, root_seed=context.root_seed,
        stream_id=context.stream_id)
end


function _transform_region_supported(model)
    model isa Union{Tilia.Standardize,Tilia.PCA,Tilia.TruncatedSVD,Tilia.Select} ||
        model isa Tilia.MinMaxScale
end

function _mixed_regions(graph)
    length(graph.nodes) > 1 || return nothing
    models = [node.model for node in graph.nodes]
    (_probabilistic_head(last(models)) || _regression_head(last(models))) || return nothing
    supported = [_transform_region_supported(model) for model in models[1:end-1]]
    push!(supported, true)
    predecessors = Tilia.graph_predecessors(graph)
    consumers = [Int[] for _ in graph.nodes]
    for (from, to) in graph.edges
        push!(consumers[from], to)
    end
    regions = UnitRange{Int}[]
    index = 1
    while index <= length(supported)
        if supported[index]
            final = index
            while final < length(supported) && supported[final + 1] &&
                  predecessors[final + 1] == [final] && consumers[final] == [final + 1] &&
                  count(i -> models[i] isa Tilia.MinMaxScale && models[i].clip,
                        index:final + 1) <= 1
                final += 1
            end
            push!(regions, index:final)
            index = final + 1
        else
            index += 1
        end
    end
    isempty(regions) || last(regions).stop == length(models) || return nothing
    length(regions) == 1 && first(regions).start == 1 && return nothing
    regions
end

function _cpu_prefix_output(cpu_graph, X, region_start)
    if cpu_graph.graph.edges != [(index, index + 1)
                                 for index in 1:length(cpu_graph.graph.nodes)-1]
        predecessors = Tilia.graph_predecessors(cpu_graph.graph)
        values = Vector{Any}(undef, length(cpu_graph.graph.nodes))
        for index in 1:region_start-1
            input = Tilia._graph_input(values, predecessors[index], X)
            values[index] = Tilia.transform(cpu_graph.fitted_nodes[index], input)
        end
        return Tilia._graph_input(values, predecessors[region_start], X)
    end
    value = X
    for index in 1:region_start-1
        value = Tilia.transform(cpu_graph.fitted_nodes[index], value)
    end
    value
end

function _apply_mixed_placement!(capabilities, regions, graph)
    numerical = capabilities.graph
    accelerator_ids = Set(Iterators.flatten(regions))
    for node in numerical.nodes
        node.device = node.semantic_node_id in accelerator_ids ? :reactant : :cpu
    end
    for primitive in numerical.primitives
        primitive.device = primitive.semantic_node_id in accelerator_ids ? :reactant : :cpu
    end
    for (from, to) in graph.edges
        from_device = from in accelerator_ids ? :reactant : :cpu
        to_device = to in accelerator_ids ? :reactant : :cpu
        from_device == to_device && continue
        operation = from_device === :cpu ? :transfer_host_to_device :
                    :transfer_device_to_host
        transfer_id = length(numerical.primitives) + 1
        transfer = Tilia.NumericalExecutionNode(
            transfer_id, to, operation, (), (), Any, :dense,
            to_device, 0, to, Int[], :owned_output)
        push!(numerical.primitives, transfer)
        push!(capabilities.primitives, (
            node_id=to, primitive_id=transfer_id,
            operation=operation, phase=:inference, supported=true, reason=""))
    end
    capabilities
end
