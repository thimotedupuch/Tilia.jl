"""Density-based clustering with deterministic expansion and label `0` for noise."""
struct DBSCAN <: AbstractEstimator
    radius::Float64
    min_neighbors::Int
    function DBSCAN(; radius::Real=0.5, min_neighbors::Integer=5)
        isfinite(radius) && radius > 0 || throw(InvalidHyperparameterError(
            "DBSCAN radius must be finite and positive."))
        min_neighbors > 0 || throw(InvalidHyperparameterError(
            "DBSCAN min_neighbors must be positive."))
        new(Float64(radius), Int(min_neighbors))
    end
end

struct FittedDBSCAN{M,T,R,S} <: AbstractFittedEstimator
    model::M
    labels::Vector{Int}
    core_indices::Vector{Int}
    training_data::Matrix{T}
    report::R
    schema::S
end

capabilities(::Type{<:DBSCAN}) = (task=:clustering, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _dbscan_region(distances, point, squared_radius)
    [candidate for candidate in axes(distances, 2)
     if distances[point, candidate] <= squared_radius]
end

function fit(model::DBSCAN, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "DBSCAN fitting")
    _validate_numeric_matrix(X, "DBSCAN")
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "DBSCAN requires observations and features."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    distances = Kernels.pairwise_distances(data; metric=:squared_euclidean)
    squared_radius = T(model.radius)^2
    core = [count(<=(squared_radius), view(distances, point, :)) >=
            model.min_neighbors for point in axes(data, 1)]
    visited = falses(size(data, 1))
    labels = zeros(Int, size(data, 1))
    cluster = 0
    queued = falses(size(data, 1))
    for point in axes(data, 1)
        visited[point] && continue
        visited[point] = true
        neighbors = _dbscan_region(distances, point, squared_radius)
        length(neighbors) >= model.min_neighbors || continue
        cluster += 1
        labels[point] = cluster
        fill!(queued, false)
        queue = copy(neighbors)
        queued[queue] .= true
        cursor = 1
        while cursor <= length(queue)
            candidate = queue[cursor]
            cursor += 1
            if !visited[candidate]
                visited[candidate] = true
                candidate_neighbors = _dbscan_region(
                    distances, candidate, squared_radius)
                if length(candidate_neighbors) >= model.min_neighbors
                    for neighbor in candidate_neighbors
                        if !queued[neighbor]
                            push!(queue, neighbor)
                            queued[neighbor] = true
                        end
                    end
                end
            end
            labels[candidate] == 0 && (labels[candidate] = cluster)
        end
    end
    core_indices = findall(core)
    details = (clusters=cluster, noise_observations=count(iszero, labels),
               core_observations=length(core_indices), radius=model.radius,
               min_neighbors=model.min_neighbors)
    FittedDBSCAN(model, labels, core_indices, data,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), infer_schema(X))
end

"""Assign new observations to the nearest reachable core cluster, or `0` as noise."""
function predict(fitted::FittedDBSCAN, X::AbstractMatrix)
    _validate_numeric_matrix(X, "DBSCAN")
    _validate_feature_count(fitted.schema, X, "DBSCAN")
    isempty(fitted.core_indices) && return zeros(Int, size(X, 1))
    core_data = view(fitted.training_data, fitted.core_indices, :)
    distances = Kernels.pairwise_distances(X, core_data; metric=:squared_euclidean)
    squared_radius = eltype(distances)(fitted.model.radius)^2
    [begin
        nearest = argmin(view(distances, row, :))
        distances[row, nearest] <= squared_radius ?
            fitted.labels[fitted.core_indices[nearest]] : 0
    end for row in axes(X, 1)]
end

report(fitted::FittedDBSCAN) = fitted.report
