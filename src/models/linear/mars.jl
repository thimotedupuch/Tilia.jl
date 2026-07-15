"""One hinge factor in a multivariate adaptive regression spline basis."""
struct HingeFactor{T}
    feature::Int
    knot::T
    direction::Int8
end

"""Product of hinge factors forming one adaptive spline basis term."""
struct MARSTerm{T}
    factors::Vector{HingeFactor{T}}
end

function MARSTerm(factors::Vector{Any})
    isempty(factors) && return MARSTerm(HingeFactor{Float64}[])
    T = typeof(first(factors).knot)
    MARSTerm(HingeFactor{T}[factor for factor in factors])
end

"""Adaptive hinge-spline regression with forward selection and GCV pruning."""
struct MARSRegressor <: AbstractPredictor
    max_terms::Int
    max_degree::Int
    max_knots::Int
    pruning_penalty::Float64
    tolerance::Float64
    function MARSRegressor(; max_terms::Integer=21, max_degree::Integer=1,
                           max_knots::Integer=32,
                           pruning_penalty::Real=3.0,
                           tolerance::Real=1e-6)
        max_terms >= 3 || throw(InvalidHyperparameterError(
            "MARSRegressor max_terms must be at least three."))
        max_degree > 0 || throw(InvalidHyperparameterError(
            "MARSRegressor max_degree must be positive."))
        max_knots > 0 || throw(InvalidHyperparameterError(
            "MARSRegressor max_knots must be positive."))
        isfinite(pruning_penalty) && pruning_penalty >= 0 ||
            throw(InvalidHyperparameterError(
                "MARSRegressor pruning_penalty must be finite and nonnegative."))
        isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
            "MARSRegressor tolerance must be finite and nonnegative."))
        new(Int(max_terms), Int(max_degree), Int(max_knots),
            Float64(pruning_penalty), Float64(tolerance))
    end
end

struct FittedMARSRegressor{M,T,R,S} <: AbstractFittedEstimator
    model::M
    terms::Vector{MARSTerm{T}}
    coefficients::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:MARSRegressor}) = (task=:regression, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=false)

function _mars_term_column(X, term::MARSTerm{T}) where {T}
    column = ones(T, size(X, 1))
    @inbounds for factor in term.factors
        if factor.direction > 0
            column .*= max.(T.(view(X, :, factor.feature)) .- factor.knot, zero(T))
        else
            column .*= max.(factor.knot .- T.(view(X, :, factor.feature)), zero(T))
        end
    end
    column
end

function _mars_design(X, terms)
    hcat((_mars_term_column(X, term) for term in terms)...)
end

function _mars_solve(design, target)
    coefficients = qr(design) \ target
    residual = target - design * coefficients
    coefficients, sum(abs2, residual)
end

function _mars_knots(values, max_knots)
    unique_values = sort!(unique(values))
    length(unique_values) <= 2 && return eltype(values)[]
    candidates = unique_values[2:end-1]
    length(candidates) <= max_knots && return candidates
    positions = unique(round.(Int, range(1, length(candidates); length=max_knots)))
    candidates[positions]
end

function _mars_gcv(rss, observations, terms, penalty)
    effective = terms + penalty * max(terms - 1, 0) / 2
    denominator = max(1 - effective / observations, eps(Float64))
    rss / observations / denominator^2
end

function fit(model::MARSRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    reject_unsupported_weights(model, weights)
    require_cpu(context, "MARSRegressor fitting")
    _validate_regression_data(X, y, nothing, "MARSRegressor")
    T = float(promote_type(eltype(X), eltype(y)))
    data, target = Matrix{T}(X), T.(y)
    terms = MARSTerm{T}[MARSTerm(HingeFactor{T}[])]
    design = ones(T, size(X, 1), 1)
    _, current_rss = _mars_solve(design, target)
    forward_rss = T[current_rss]
    while length(terms) + 2 <= model.max_terms
        best = nothing
        for parent_index in eachindex(terms)
            parent = terms[parent_index]
            length(parent.factors) < model.max_degree || continue
            parent_column = view(design, :, parent_index)
            used_features = Set(factor.feature for factor in parent.factors)
            for feature in axes(data, 2)
                feature in used_features && continue
                for knot in _mars_knots(view(data, :, feature), model.max_knots)
                    right = parent_column .* max.(view(data, :, feature) .- knot, zero(T))
                    left = parent_column .* max.(knot .- view(data, :, feature), zero(T))
                    count(value -> !iszero(value), right) > 1 &&
                        count(value -> !iszero(value), left) > 1 || continue
                    candidate_design = hcat(design, right, left)
                    _, rss = _mars_solve(candidate_design, target)
                    if best === nothing || rss < best.rss
                        best = (rss=rss, parent=parent_index, feature=feature,
                                knot=T(knot), right=right, left=left)
                    end
                end
            end
        end
        best === nothing && break
        improvement = current_rss - best.rss
        improvement > T(model.tolerance) * max(current_rss, one(T)) || break
        parent_factors = terms[best.parent].factors
        push!(terms, MARSTerm(vcat(parent_factors,
            HingeFactor{T}(best.feature, best.knot, Int8(1)))))
        push!(terms, MARSTerm(vcat(parent_factors,
            HingeFactor{T}(best.feature, best.knot, Int8(-1)))))
        design = hcat(design, best.right, best.left)
        current_rss = best.rss
        push!(forward_rss, current_rss)
    end
    active = collect(eachindex(terms))
    best_active = copy(active)
    _, full_rss = _mars_solve(design, target)
    best_gcv = _mars_gcv(full_rss, size(X, 1), length(active), model.pruning_penalty)
    while length(active) > 1
        removal = nothing
        for term_index in active[2:end]
            candidate = filter(!=(term_index), active)
            _, rss = _mars_solve(view(design, :, candidate), target)
            gcv = _mars_gcv(rss, size(X, 1), length(candidate), model.pruning_penalty)
            (removal === nothing || gcv < removal.gcv) &&
                (removal = (term=term_index, gcv=gcv, active=candidate))
        end
        active = removal.active
        if removal.gcv < best_gcv
            best_gcv = removal.gcv
            best_active = copy(active)
        end
    end
    selected_terms = terms[best_active]
    selected_design = view(design, :, best_active)
    coefficients, rss = _mars_solve(selected_design, target)
    details = (forward_terms=length(terms), selected_terms=length(selected_terms),
               forward_rss=forward_rss, residual_sum_squares=rss,
               gcv=best_gcv, max_degree=model.max_degree, solver=:qr)
    FittedMARSRegressor(model, selected_terms, coefficients,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context), with_target(infer_schema(X), y))
end

function predict(fitted::FittedMARSRegressor, X::AbstractMatrix)
    _validate_numeric_matrix(X, "MARSRegressor")
    _validate_feature_count(fitted.schema, X, "MARSRegressor")
    _mars_design(X, fitted.terms) * fitted.coefficients
end

report(fitted::FittedMARSRegressor) = fitted.report
