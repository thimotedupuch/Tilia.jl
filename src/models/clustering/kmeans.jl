"""Lloyd k-means with random or k-means++ initialization and deterministic restarts."""
struct KMeans <: AbstractEstimator
    n_clusters::Int
    init::Symbol
    n_init::Int
    max_iterations::Int
    tolerance::Float64
    function KMeans(; n_clusters::Integer=8, init::Symbol=:kmeanspp,
                    n_init::Integer=10, max_iterations::Integer=300,
                    tolerance::Real=1e-4)
        n_clusters > 0 || throw(InvalidHyperparameterError("KMeans n_clusters must be positive."))
        init in (:kmeanspp, :random) || throw(InvalidHyperparameterError(
            "KMeans init must be :kmeanspp or :random."))
        n_init > 0 || throw(InvalidHyperparameterError("KMeans n_init must be positive."))
        max_iterations > 0 || throw(InvalidHyperparameterError("KMeans max_iterations must be positive."))
        isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
            "KMeans tolerance must be finite and nonnegative."))
        new(Int(n_clusters), init, Int(n_init), Int(max_iterations), Float64(tolerance))
    end
end

struct FittedKMeans{M,T,R,S} <: AbstractFittedEstimator
    model::M
    centers::Matrix{T}
    labels::Vector{Int}
    inertia::T
    iterations::Int
    converged::Bool
    report::R
    schema::S
end

capabilities(::Type{<:KMeans}) = (task=:clustering, sparse=false, missing=false,
    weights=false, partial_fit=false, probabilistic=false)

function _squared_distance_matrix(X, centers)
    Kernels.pairwise_distances(X, centers; metric=:squared_euclidean)
end

function _sample_weighted(rng, weights)
    total = sum(weights)
    iszero(total) && return rand(rng, eachindex(weights))
    threshold = rand(rng) * total
    cumulative = zero(total)
    for index in eachindex(weights)
        cumulative += weights[index]
        cumulative >= threshold && return index
    end
    lastindex(weights)
end

function _initial_centers(model, X, rng)
    n = size(X, 1)
    if model.init === :random
        return copy(X[randperm(rng, n)[1:model.n_clusters], :])
    end
    indices = Vector{Int}(undef, model.n_clusters)
    indices[1] = rand(rng, 1:n)
    closest = [sum(abs2, view(X, row, :) .- view(X, indices[1], :)) for row in 1:n]
    for cluster in 2:model.n_clusters
        indices[cluster] = _sample_weighted(rng, closest)
        for row in 1:n
            distance = sum(abs2, view(X, row, :) .- view(X, indices[cluster], :))
            closest[row] = min(closest[row], distance)
        end
    end
    copy(X[indices, :])
end

function _kmeans_run(model, X, rng, tolerance, max_iterations)
    centers = _initial_centers(model, X, rng)
    labels = ones(Int, size(X, 1))
    converged = false
    iterations = max_iterations
    objective_history = eltype(X)[]
    for iteration in 1:max_iterations
        distances = _squared_distance_matrix(X, centers)
        labels .= map(row -> argmin(view(distances, row, :)), axes(X, 1))
        new_centers = similar(centers)
        for cluster in 1:model.n_clusters
            members = findall(==(cluster), labels)
            if isempty(members)
                nearest = vec(minimum(distances; dims=2))
                new_centers[cluster, :] .= view(X, argmax(nearest), :)
            else
                new_centers[cluster, :] .= vec(mean(view(X, members, :); dims=1))
            end
        end
        shift = maximum(sum(abs2, view(new_centers, cluster, :) .-
                                  view(centers, cluster, :)) for cluster in 1:model.n_clusters)
        centers = new_centers
        updated_distances = _squared_distance_matrix(X, centers)
        push!(objective_history,
            sum(updated_distances[row, labels[row]] for row in axes(X, 1)))
        if shift <= tolerance^2
            converged = true
            iterations = iteration
            break
        end
    end
    distances = _squared_distance_matrix(X, centers)
    labels .= map(row -> argmin(view(distances, row, :)), axes(X, 1))
    inertia = sum(distances[row, labels[row]] for row in axes(X, 1))
    (centers=centers, labels=labels, inertia=inertia, iterations=iterations,
     converged=converged, objective_history=objective_history)
end

function fit(model::KMeans, X::AbstractMatrix; context=default_context())
    require_cpu(context, "KMeans fitting")
    _validate_numeric_matrix(X, "KMeans")
    n, p = size(X)
    p > 0 || throw(UnsupportedDataError("KMeans requires at least one feature."))
    n >= model.n_clusters || throw(UnsupportedDataError(
        "KMeans requires at least n_clusters observations; received $n for $(model.n_clusters) clusters."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    tolerance = T(effective_tolerance(context, model.tolerance))
    max_iterations = effective_max_iterations(context, model.max_iterations)
    best = nothing
    for restart in 1:model.n_init
        restart_context = derive_context(context, :kmeans, :restart, restart)
        candidate = _kmeans_run(model, data, restart_context.rng,
                                tolerance, max_iterations)
        (best === nothing || candidate.inertia < best.inertia) && (best = candidate)
    end
    warnings = best.converged ? String[] : ["KMeans reached max_iterations without convergence."]
    details = (inertia=best.inertia, iterations=best.iterations, converged=best.converged,
               objective_history=best.objective_history, n_clusters=model.n_clusters,
               n_init=model.n_init, init=model.init)
    fit_report = FitReport(observations=n, features=p, backend=:cpu,
        warnings=warnings, details=details, context=context)
    FittedKMeans(model, best.centers, best.labels, best.inertia, best.iterations, best.converged,
        fit_report, infer_schema(X))
end

function transform(fitted::FittedKMeans, X::AbstractMatrix)
    _validate_numeric_matrix(X, "KMeans")
    _validate_feature_count(fitted.schema, X, "KMeans")
    sqrt.(_squared_distance_matrix(Matrix{eltype(fitted.centers)}(X), fitted.centers))
end

function predict(fitted::FittedKMeans, X::AbstractMatrix)
    distances = transform(fitted, X)
    map(row -> argmin(view(distances, row, :)), axes(distances, 1))
end

report(fitted::FittedKMeans) = fitted.report
