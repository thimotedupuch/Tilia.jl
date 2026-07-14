struct DeviceAssignment
    node_id::Int
    device::Symbol
end

struct DeviceTransfer
    from_node::Int
    to_node::Int
    from_device::Symbol
    to_device::Symbol
end

struct PlacementPlan
    assignments::Vector{DeviceAssignment}
    transfers::Vector{DeviceTransfer}
end

"""Remove duplicate adjacent transfer declarations."""
function coalesce_transfers(transfers::Vector{DeviceTransfer})
    result = DeviceTransfer[]
    for transfer in transfers
        isempty(result) || transfer == last(result) || push!(result, transfer)
        isempty(result) && push!(result, transfer)
    end
    result
end

"""Assign graph nodes to explicit devices and identify required edge transfers."""
function device_placement(graph::SemanticGraph; default::Symbol=:cpu,
                          overrides=Dict{Int,Symbol}())
    default in (:cpu, :reactant) || throw(GraphValidationError(
        "default graph device must be :cpu or :reactant."))
    assignments = DeviceAssignment[]
    devices = Dict{Int,Symbol}()
    for node in graph.nodes
        device = get(overrides, node.id, default)
        device in (:cpu, :reactant) || throw(GraphValidationError(
            "node $(node.id) has unsupported device $device."))
        devices[node.id] = device
        push!(assignments, DeviceAssignment(node.id, device))
    end
    all(id -> 1 <= id <= length(graph.nodes), keys(overrides)) || throw(GraphValidationError(
        "device override refers to an unknown graph node."))
    transfers = DeviceTransfer[DeviceTransfer(from, to, devices[from], devices[to])
        for (from, to) in graph.edges if devices[from] != devices[to]]
    PlacementPlan(assignments, coalesce_transfers(transfers))
end
