"""Deterministic agglomerative clustering with single, complete, or average linkage."""
struct AgglomerativeClustering <: AbstractEstimator
    n_clusters::Int
    linkage::Symbol
    function AgglomerativeClustering(; n_clusters::Integer=2,
                                     linkage::Symbol=:average)
        n_clusters > 0 || throw(InvalidHyperparameterError(
            "AgglomerativeClustering n_clusters must be positive."))
        linkage in (:single, :complete, :average) || throw(InvalidHyperparameterError(
            "AgglomerativeClustering linkage must be :single, :complete, or :average."))
        new(Int(n_clusters), linkage)
    end
end

struct FittedAgglomerativeClustering{M,T,R,S} <: AbstractFittedEstimator
    model::M
    labels::Vector{Int}
    centers::Matrix{T}
    children::Matrix{Int}
    merge_distances::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:AgglomerativeClustering}) = (task=:clustering,
    sparse=false, missing=false, weights=false, partial_fit=false,
    probabilistic=false)

function _heap_push!(heap, item)
    push!(heap, item)
    position = length(heap)
    while position > 1
        parent = position >>> 1
        isless(heap[position], heap[parent]) || break
        heap[position], heap[parent] = heap[parent], heap[position]
        position = parent
    end
    heap
end

function _heap_pop!(heap)
    first_item = first(heap)
    last_item = pop!(heap)
    isempty(heap) && return first_item
    heap[1] = last_item
    position = 1
    while true
        left = position << 1
        left > length(heap) && break
        right = left + 1
        child = right <= length(heap) && isless(heap[right], heap[left]) ? right : left
        isless(heap[child], heap[position]) || break
        heap[position], heap[child] = heap[child], heap[position]
        position = child
    end
    first_item
end

function _linkage_distance(linkage, left_distance, right_distance,
                           left_size, right_size)
    linkage === :single && return min(left_distance, right_distance)
    linkage === :complete && return max(left_distance, right_distance)
    (left_size * left_distance + right_size * right_distance) /
        (left_size + right_size)
end

function fit(model::AgglomerativeClustering, X::AbstractMatrix; weights=nothing,
             context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "AgglomerativeClustering fitting")
    _validate_numeric_matrix(X, "AgglomerativeClustering")
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError(
        "AgglomerativeClustering requires observations and features."))
    model.n_clusters <= n || throw(UnsupportedDataError(
        "AgglomerativeClustering n_clusters cannot exceed the observation count."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    capacity = 2n - 1
    distances = fill(T(Inf), capacity, capacity)
    initial = Kernels.pairwise_distances(data; metric=:euclidean)
    distances[1:n, 1:n] .= initial
    heap = Tuple{T,Int,Int}[]
    sizehint!(heap, n * (n - 1) ÷ 2)
    for right in 2:n, left in 1:right-1
        _heap_push!(heap, (initial[left, right], left, right))
    end
    active = falses(capacity)
    active[1:n] .= true
    cluster_sizes = zeros(Int, capacity)
    cluster_sizes[1:n] .= 1
    members = [Int[] for _ in 1:capacity]
    for index in 1:n
        members[index] = [index]
    end
    merges = n - model.n_clusters
    children = Matrix{Int}(undef, merges, 2)
    merge_distances = Vector{T}(undef, merges)
    next_cluster = n
    for merge_index in 1:merges
        candidate = _heap_pop!(heap)
        while !(active[candidate[2]] && active[candidate[3]])
            candidate = _heap_pop!(heap)
        end
        merge_distance, left, right = candidate
        next_cluster += 1
        children[merge_index, :] .= (left, right)
        merge_distances[merge_index] = merge_distance
        active[left] = false
        active[right] = false
        active[next_cluster] = true
        left_size, right_size = cluster_sizes[left], cluster_sizes[right]
        cluster_sizes[next_cluster] = left_size + right_size
        members[next_cluster] = vcat(members[left], members[right])
        for other in findall(active)
            other == next_cluster && continue
            distance = _linkage_distance(model.linkage,
                distances[left, other], distances[right, other],
                left_size, right_size)
            distances[next_cluster, other] = distance
            distances[other, next_cluster] = distance
            a, b = minmax(other, next_cluster)
            _heap_push!(heap, (distance, a, b))
        end
    end
    final_clusters = findall(active)
    sort!(final_clusters; by=cluster -> minimum(members[cluster]))
    labels = zeros(Int, n)
    centers = Matrix{T}(undef, length(final_clusters), p)
    for (label, cluster) in enumerate(final_clusters)
        labels[members[cluster]] .= label
        centers[label, :] .= vec(mean(view(data, members[cluster], :); dims=1))
    end
    details = (clusters=length(final_clusters), linkage=model.linkage,
               merges=merges, quadratic_memory=true)
    FittedAgglomerativeClustering(model, labels, centers, children,
        merge_distances, FitReport(observations=n, features=p,
            details=details, context=context), infer_schema(X))
end

"""Assign new observations to the nearest final-cluster centroid."""
function predict(fitted::FittedAgglomerativeClustering, X::AbstractMatrix)
    _validate_numeric_matrix(X, "AgglomerativeClustering")
    _validate_feature_count(fitted.schema, X, "AgglomerativeClustering")
    distances = Kernels.pairwise_distances(X, fitted.centers;
                                           metric=:squared_euclidean)
    [argmin(view(distances, row, :)) for row in axes(distances, 1)]
end

report(fitted::FittedAgglomerativeClustering) = fitted.report
