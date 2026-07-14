struct IterativeResult{P,T,H}
    parameters::P
    objective_history::H
    iterations::Int
    converged::Bool
    residual_norm::T
    solver::Symbol
end

_apply_operator(operator, vector) = operator isa AbstractMatrix ? operator * vector : operator(vector)

"""Conjugate-gradient solve for a symmetric positive-definite operator."""
function conjugate_gradient(operator, right_hand_side::AbstractVector;
        initial=zeros(eltype(right_hand_side), length(right_hand_side)),
        tolerance=sqrt(eps(float(eltype(right_hand_side)))),
        max_iterations::Integer=max(20, 10length(right_hand_side)))
    parameters = copy(initial)
    residual = right_hand_side .- _apply_operator(operator, parameters)
    direction = copy(residual)
    squared_residual = dot(residual, residual)
    history = [sqrt(squared_residual)]
    converged = history[end] <= tolerance
    iterations = 0
    for iteration in 1:max_iterations
        converged && break
        product = _apply_operator(operator, direction)
        curvature = dot(direction, product)
        curvature > 0 || throw(ArgumentError("conjugate gradient requires positive curvature."))
        step = squared_residual / curvature
        parameters .+= step .* direction
        residual .-= step .* product
        new_squared_residual = dot(residual, residual)
        push!(history, sqrt(new_squared_residual))
        iterations = iteration
        converged = history[end] <= tolerance
        direction .= residual .+ (new_squared_residual / squared_residual) .* direction
        squared_residual = new_squared_residual
    end
    IterativeResult(parameters, history, iterations, converged, last(history), :conjugate_gradient)
end

"""LSQR solve for rectangular least-squares systems."""
function lsqr(A::AbstractMatrix, b::AbstractVector; tolerance=1e-8,
              max_iterations::Integer=2min(size(A)...))
    size(A, 1) == length(b) || throw(DimensionMismatch("LSQR rows must match right-hand side."))
    T = float(promote_type(eltype(A), eltype(b)))
    x = zeros(T, size(A, 2))
    u = T.(b)
    beta = norm(u)
    iszero(beta) && return IterativeResult(x, T[zero(T)], 0, true, zero(T), :lsqr)
    u ./= beta
    v = transpose(A) * u
    alpha = norm(v)
    iszero(alpha) && return IterativeResult(x, T[beta], 0, false, beta, :lsqr)
    v ./= alpha
    w = copy(v)
    phi_bar, rho_bar = beta, alpha
    history = T[beta]
    converged = false
    iterations = max_iterations
    for iteration in 1:max_iterations
        u = A * v .- alpha .* u
        beta = norm(u)
        iszero(beta) || (u ./= beta)
        v = transpose(A) * u .- beta .* v
        alpha = norm(v)
        iszero(alpha) || (v ./= alpha)
        rho = hypot(rho_bar, beta)
        cosine, sine = rho_bar / rho, beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        x .+= (phi / rho) .* w
        w .= v .- (theta / rho) .* w
        residual = norm(A * x - b)
        push!(history, residual)
        if residual <= tolerance
            converged = true
            iterations = iteration
            break
        end
    end
    IterativeResult(x, history, iterations, converged, last(history), :lsqr)
end

function _backtracking(objective, parameters, direction, gradient, value)
    step = one(eltype(parameters))
    slope = dot(gradient, direction)
    while objective(parameters .+ step .* direction) > value + eltype(parameters)(1e-4) * step * slope
        step *= eltype(parameters)(0.5)
        step > eps(eltype(parameters)) || break
    end
    step
end

"""Limited-memory BFGS with Armijo backtracking."""
function lbfgs(objective, gradient!, initial::AbstractVector; memory::Integer=10,
               tolerance=1e-6, max_iterations::Integer=500)
    memory > 0 || throw(ArgumentError("L-BFGS memory must be positive."))
    parameters = float.(copy(initial))
    gradient = similar(parameters)
    gradient!(gradient, parameters)
    value = objective(parameters)
    history = [value]
    steps, changes, inverse_curvatures = Vector{typeof(parameters)}(), Vector{typeof(parameters)}(), eltype(parameters)[]
    converged = norm(gradient, Inf) <= tolerance
    iterations = 0
    for iteration in 1:max_iterations
        converged && break
        direction = copy(gradient)
        alphas = eltype(parameters)[]
        for index in length(steps):-1:1
            alpha = inverse_curvatures[index] * dot(steps[index], direction)
            push!(alphas, alpha)
            direction .-= alpha .* changes[index]
        end
        scale = isempty(steps) ? one(eltype(parameters)) :
            dot(steps[end], changes[end]) / dot(changes[end], changes[end])
        direction .*= scale
        for index in eachindex(steps)
            beta = inverse_curvatures[index] * dot(changes[index], direction)
            alpha = alphas[length(steps) - index + 1]
            direction .+= steps[index] .* (alpha - beta)
        end
        direction .*= -1
        step_size = _backtracking(objective, parameters, direction, gradient, value)
        new_parameters = parameters .+ step_size .* direction
        new_gradient = similar(gradient)
        gradient!(new_gradient, new_parameters)
        parameter_step = new_parameters .- parameters
        gradient_change = new_gradient .- gradient
        curvature = dot(parameter_step, gradient_change)
        if curvature > eps(eltype(parameters))
            length(steps) == memory && (popfirst!(steps); popfirst!(changes); popfirst!(inverse_curvatures))
            push!(steps, parameter_step); push!(changes, gradient_change); push!(inverse_curvatures, inv(curvature))
        end
        parameters, gradient = new_parameters, new_gradient
        value = objective(parameters)
        push!(history, value)
        iterations = iteration
        converged = norm(gradient, Inf) <= tolerance
    end
    IterativeResult(parameters, history, iterations, converged, norm(gradient), :lbfgs)
end

function _proximal_solver(solver, smooth_objective, gradient!, prox!, initial;
                          step_size, tolerance, max_iterations)
    parameters = float.(copy(initial))
    extrapolated = copy(parameters)
    momentum = one(eltype(parameters))
    gradient = similar(parameters)
    history = eltype(parameters)[]
    converged = false
    iterations = max_iterations
    for iteration in 1:max_iterations
        gradient!(gradient, extrapolated)
        updated = similar(parameters)
        prox!(updated, extrapolated .- step_size .* gradient, step_size)
        push!(history, smooth_objective(updated))
        update_norm = norm(updated .- parameters, Inf)
        if solver === :fista
            new_momentum = (one(momentum) + sqrt(one(momentum) + 4momentum^2)) / 2
            extrapolated = updated .+ ((momentum - one(momentum)) / new_momentum) .* (updated .- parameters)
            momentum = new_momentum
        else
            extrapolated = copy(updated)
        end
        parameters = updated
        if update_norm <= tolerance
            converged = true
            iterations = iteration
            break
        end
    end
    IterativeResult(parameters, history, iterations, converged,
                    isempty(history) ? zero(eltype(parameters)) : last(history), solver)
end

proximal_gradient(objective, gradient!, prox!, initial; step_size=0.01,
                  tolerance=1e-6, max_iterations=1_000) =
    _proximal_solver(:proximal_gradient, objective, gradient!, prox!, initial;
        step_size=step_size, tolerance=tolerance, max_iterations=max_iterations)

fista(objective, gradient!, prox!, initial; step_size=0.01,
      tolerance=1e-6, max_iterations=1_000) =
    _proximal_solver(:fista, objective, gradient!, prox!, initial;
        step_size=step_size, tolerance=tolerance, max_iterations=max_iterations)

"""Newton-CG optimization using a user-supplied Hessian-vector product."""
function newton_cg(objective, gradient!, hessian_vector, initial;
                   tolerance=1e-6, max_iterations=100, cg_iterations=100)
    parameters = float.(copy(initial))
    gradient = similar(parameters)
    history = [objective(parameters)]
    converged = false
    iterations = max_iterations
    for iteration in 1:max_iterations
        gradient!(gradient, parameters)
        if norm(gradient, Inf) <= tolerance
            converged = true; iterations = iteration - 1; break
        end
        operator = vector -> hessian_vector(parameters, vector)
        direction = conjugate_gradient(operator, -gradient;
            tolerance=tolerance / 10, max_iterations=cg_iterations).parameters
        step = _backtracking(objective, parameters, direction, gradient, last(history))
        parameters .+= step .* direction
        push!(history, objective(parameters))
    end
    gradient!(gradient, parameters)
    IterativeResult(parameters, history, iterations, converged, norm(gradient), :newton_cg)
end

"""Deterministic shuffled stochastic-gradient optimization."""
function stochastic_gradient(gradient_sample!, initial, observations::Integer;
        learning_rate=0.01, epochs::Integer=10, rng=Random.Xoshiro(0), objective=nothing)
    observations > 0 || throw(ArgumentError("stochastic-gradient observations must be positive."))
    parameters = float.(copy(initial))
    gradient = similar(parameters)
    history = eltype(parameters)[]
    for _ in 1:epochs
        for index in randperm(rng, observations)
            gradient_sample!(gradient, parameters, index)
            parameters .-= learning_rate .* gradient
        end
        objective === nothing || push!(history, objective(parameters))
    end
    residual = isempty(history) ? zero(eltype(parameters)) : last(history)
    IterativeResult(parameters, history, epochs, true, residual, :stochastic_gradient)
end

"""Generic expectation-maximization fixed-point driver."""
function expectation_maximization(expectation, maximization, log_likelihood, initial;
                                  tolerance=1e-6, max_iterations::Integer=100)
    parameters = initial
    history = [log_likelihood(parameters)]
    converged = false
    iterations = max_iterations
    for iteration in 1:max_iterations
        parameters = maximization(expectation(parameters))
        push!(history, log_likelihood(parameters))
        if abs(history[end] - history[end - 1]) <= tolerance * max(abs(history[end - 1]), 1)
            converged = true; iterations = iteration; break
        end
    end
    IterativeResult(parameters, history, iterations, converged,
                    abs(history[end] - history[max(1, end - 1)]), :expectation_maximization)
end
