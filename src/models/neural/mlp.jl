abstract type AbstractMLP <: AbstractPredictor end

struct MLPClassifier <: AbstractMLP
    hidden_units::Int
    activation::Symbol
    learning_rate::Float64
    l2::Float64
    max_iterations::Int
    tolerance::Float64
    function MLPClassifier(; hidden_units::Integer=32, activation::Symbol=:relu,
            learning_rate::Real=0.01, l2::Real=0.0,
            max_iterations::Integer=200, tolerance::Real=1e-6)
        _validate_mlp_parameters(hidden_units, activation, learning_rate, l2,
                                 max_iterations, tolerance)
        new(Int(hidden_units), activation, Float64(learning_rate), Float64(l2),
            Int(max_iterations), Float64(tolerance))
    end
end

struct MLPRegressor <: AbstractMLP
    hidden_units::Int
    activation::Symbol
    learning_rate::Float64
    l2::Float64
    max_iterations::Int
    tolerance::Float64
    function MLPRegressor(; hidden_units::Integer=32, activation::Symbol=:relu,
            learning_rate::Real=0.01, l2::Real=0.0,
            max_iterations::Integer=200, tolerance::Real=1e-6)
        _validate_mlp_parameters(hidden_units, activation, learning_rate, l2,
                                 max_iterations, tolerance)
        new(Int(hidden_units), activation, Float64(learning_rate), Float64(l2),
            Int(max_iterations), Float64(tolerance))
    end
end

function _validate_mlp_parameters(hidden_units, activation, learning_rate, l2,
                                  max_iterations, tolerance)
    hidden_units > 0 || throw(InvalidHyperparameterError("hidden_units must be positive."))
    activation in (:relu, :tanh) || throw(InvalidHyperparameterError(
        "activation must be :relu or :tanh."))
    isfinite(learning_rate) && learning_rate > 0 || throw(InvalidHyperparameterError(
        "learning_rate must be finite and positive."))
    isfinite(l2) && l2 >= 0 || throw(InvalidHyperparameterError("l2 must be finite and nonnegative."))
    max_iterations > 0 || throw(InvalidHyperparameterError("max_iterations must be positive."))
    isfinite(tolerance) && tolerance >= 0 || throw(InvalidHyperparameterError(
        "tolerance must be finite and nonnegative."))
end

struct FittedMLP{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    input_weights::Matrix{T}
    hidden_bias::Vector{T}
    output_weights::Matrix{T}
    output_bias::Vector{T}
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:MLPClassifier}) = (task=:classification, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=true)
capabilities(::Type{<:MLPRegressor}) = (task=:regression, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=false)

_mlp_activation(values, ::Val{:relu}) = max.(values, zero(eltype(values)))
_mlp_activation(values, ::Val{:tanh}) = tanh.(values)
_mlp_derivative(values, ::Val{:relu}) = values .> zero(eltype(values))
_mlp_derivative(values, ::Val{:tanh}) = one(eltype(values)) .- tanh.(values) .^ 2

function _fit_mlp(model, X, targets, classes, weights, context)
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_numeric_matrix(X, name)
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError("$name requires observations and features."))
    size(targets, 1) == n || throw(SchemaMismatchError("$name target row count must match X."))
    T = float(promote_type(eltype(X), eltype(targets),
                           weights === nothing ? eltype(X) : eltype(weights)))
    data, target_matrix = Matrix{T}(X), Matrix{T}(targets)
    observation_weights = _boosting_weights(weights, n, T, name)
    output_count = size(target_matrix, 2)
    initialization = derive_context(context, :mlp, :initialization)
    input_weights = randn(initialization.rng, T, p, model.hidden_units) .* sqrt(T(2 / (p + model.hidden_units)))
    hidden_bias = zeros(T, model.hidden_units)
    output_weights = randn(initialization.rng, T, model.hidden_units, output_count) .*
                     sqrt(T(2 / (model.hidden_units + output_count)))
    output_bias = zeros(T, output_count)
    activation = Val(model.activation)
    history = T[]
    converged = false
    max_iterations = effective_max_iterations(context, model.max_iterations)
    tolerance = T(effective_tolerance(context, model.tolerance))
    iterations = max_iterations
    for iteration in 1:max_iterations
        hidden_linear = data * input_weights .+ transpose(hidden_bias)
        hidden = _mlp_activation(hidden_linear, activation)
        outputs = hidden * output_weights .+ transpose(output_bias)
        if model isa MLPClassifier
            probabilities = Kernels.softmax(outputs; dims=2)
            clipped = clamp.(probabilities, eps(T), one(T))
            loss = -sum(observation_weights .* vec(sum(target_matrix .* log.(clipped); dims=2))) /
                   sum(observation_weights)
            output_delta = (probabilities .- target_matrix) .* reshape(observation_weights, :, 1) ./ sum(observation_weights)
        else
            residual = outputs .- target_matrix
            loss = sum(observation_weights .* vec(sum(abs2, residual; dims=2))) /
                   (T(2) * sum(observation_weights))
            output_delta = residual .* reshape(observation_weights, :, 1) ./ sum(observation_weights)
        end
        loss += T(model.l2) * (sum(abs2, input_weights) + sum(abs2, output_weights)) / T(2)
        push!(history, loss)
        output_gradient = transpose(hidden) * output_delta .+ T(model.l2) .* output_weights
        output_bias_gradient = vec(sum(output_delta; dims=1))
        hidden_delta = (output_delta * transpose(output_weights)) .*
                       _mlp_derivative(hidden_linear, activation)
        input_gradient = transpose(data) * hidden_delta .+ T(model.l2) .* input_weights
        hidden_bias_gradient = vec(sum(hidden_delta; dims=1))
        input_weights .-= T(model.learning_rate) .* input_gradient
        hidden_bias .-= T(model.learning_rate) .* hidden_bias_gradient
        output_weights .-= T(model.learning_rate) .* output_gradient
        output_bias .-= T(model.learning_rate) .* output_bias_gradient
        if length(history) > 1 && abs(history[end] - history[end - 1]) <= tolerance
            converged = true
            iterations = iteration
            break
        end
    end
    details = (optimizer=:batch_gradient_descent, activation=model.activation,
               hidden_units=model.hidden_units, iterations=iterations,
               converged=converged, objective_history=history, l2=model.l2)
    schema = infer_schema(X)
    schema = classes === nothing ? with_target(schema, vec(targets)) :
             with_class_target(schema, classes)
    FittedMLP(model, input_weights, hidden_bias, output_weights, output_bias, classes,
        FitReport(status=converged ? :success : :max_iterations,
            observations=n, features=p, backend=:cpu,
            warnings=converged ? String[] : ["$name reached max_iterations."],
            details=details, context=context), schema)
end

function fit(model::MLPClassifier, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "MLPClassifier target has length $(length(y)); expected $(size(X, 1))."))
    classes = _classification_classes(y)
    targets = zeros(float(eltype(X)), length(y), length(classes))
    for row in eachindex(y)
        targets[row, searchsortedfirst(classes, y[row])] = 1
    end
    _fit_mlp(model, X, targets, classes, weights, context)
end

function fit(model::MLPRegressor, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    _validate_regression_data(X, y, weights, "MLPRegressor")
    _fit_mlp(model, X, reshape(y, :, 1), nothing, weights, context)
end

function _mlp_outputs(fitted, X)
    hidden = _mlp_activation(X * fitted.input_weights .+ transpose(fitted.hidden_bias),
                             Val(fitted.model.activation))
    hidden * fitted.output_weights .+ transpose(fitted.output_bias)
end

function predict_proba(fitted::FittedMLP{<:MLPClassifier}, X::AbstractMatrix)
    _validate_numeric_matrix(X, "MLPClassifier")
    _validate_feature_count(fitted.schema, X, "MLPClassifier")
    Kernels.softmax(_mlp_outputs(fitted, X); dims=2)
end

function predict(fitted::FittedMLP{<:MLPClassifier}, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(X, 1)]
end

function predict(fitted::FittedMLP{<:MLPRegressor}, X::AbstractMatrix)
    _validate_numeric_matrix(X, "MLPRegressor")
    _validate_feature_count(fitted.schema, X, "MLPRegressor")
    vec(_mlp_outputs(fitted, X))
end

report(fitted::FittedMLP) = fitted.report
