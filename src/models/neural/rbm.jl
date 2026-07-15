"""Bernoulli restricted Boltzmann machine trained with mini-batch CD-1."""
struct BernoulliRBM <: AbstractTransformer
    n_components::Int
    learning_rate::Float64
    batch_size::Int
    n_iterations::Int
    function BernoulliRBM(; n_components::Integer=256, learning_rate::Real=0.1,
            batch_size::Integer=10, n_iterations::Integer=10)
        n_components > 0 || throw(InvalidHyperparameterError("n_components must be positive."))
        isfinite(learning_rate) && learning_rate > 0 || throw(InvalidHyperparameterError(
            "learning_rate must be finite and positive."))
        batch_size > 0 || throw(InvalidHyperparameterError("batch_size must be positive."))
        n_iterations > 0 || throw(InvalidHyperparameterError("n_iterations must be positive."))
        new(Int(n_components), Float64(learning_rate), Int(batch_size), Int(n_iterations))
    end
end

struct FittedBernoulliRBM{M,T,R,S} <: AbstractFittedTransformer
    model::M
    weights::Matrix{T}
    visible_bias::Vector{T}
    hidden_bias::Vector{T}
    report::R
    schema::S
end

capabilities(::Type{<:BernoulliRBM}) = (task=:transformation, sparse=false,
    missing=false, weights=false, partial_fit=false, probabilistic=true)

function _validate_bernoulli_data(X, name)
    _validate_numeric_matrix(X, name)
    all(value -> 0 <= value <= 1, X) || throw(UnsupportedDataError(
        "$name requires feature values in [0, 1]."))
end

function fit(model::BernoulliRBM, X::AbstractMatrix; context=default_context())
    require_cpu(context, "BernoulliRBM fitting")
    _validate_bernoulli_data(X, "BernoulliRBM")
    n, p = size(X)
    n > 0 && p > 0 || throw(UnsupportedDataError("BernoulliRBM requires observations and features."))
    T = float(eltype(X))
    data = Matrix{T}(X)
    initialization = derive_context(context, :rbm, :initialization)
    weights = randn(initialization.rng, T, p, model.n_components) .* T(0.01)
    visible_bias = zeros(T, p)
    hidden_bias = zeros(T, model.n_components)
    history = T[]
    for iteration in 1:model.n_iterations
        iteration_context = derive_context(context, :rbm, :iteration, iteration)
        ordering = randperm(iteration_context.rng, n)
        for start in 1:model.batch_size:n
            indices = ordering[start:min(start + model.batch_size - 1, n)]
            visible = view(data, indices, :)
            hidden_probabilities = Kernels.sigmoid(visible * weights .+ transpose(hidden_bias))
            hidden_states = T.(rand(iteration_context.rng, T, size(hidden_probabilities)) .< hidden_probabilities)
            visible_probabilities = Kernels.sigmoid(hidden_states * transpose(weights) .+
                                                     transpose(visible_bias))
            negative_hidden = Kernels.sigmoid(visible_probabilities * weights .+
                                              transpose(hidden_bias))
            scale = T(model.learning_rate / length(indices))
            weights .+= scale .* (transpose(visible) * hidden_probabilities .-
                                  transpose(visible_probabilities) * negative_hidden)
            visible_bias .+= T(model.learning_rate) .* vec(mean(visible .- visible_probabilities; dims=1))
            hidden_bias .+= T(model.learning_rate) .* vec(mean(hidden_probabilities .- negative_hidden; dims=1))
        end
        hidden = Kernels.sigmoid(data * weights .+ transpose(hidden_bias))
        reconstruction = Kernels.sigmoid(hidden * transpose(weights) .+ transpose(visible_bias))
        push!(history, mean(abs2, data .- reconstruction))
    end
    details = (algorithm=:contrastive_divergence_1, n_components=model.n_components,
               iterations=model.n_iterations, batch_size=min(model.batch_size, n),
               reconstruction_error=last(history), objective_history=history)
    FittedBernoulliRBM(model, weights, visible_bias, hidden_bias,
        FitReport(observations=n, features=p, backend=:cpu, details=details,
                  context=context), infer_schema(X))
end

function transform(fitted::FittedBernoulliRBM, X::AbstractMatrix)
    _validate_bernoulli_data(X, "BernoulliRBM")
    _validate_feature_count(fitted.schema, X, "BernoulliRBM")
    Kernels.sigmoid(X * fitted.weights .+ transpose(fitted.hidden_bias))
end

function inverse_transform(fitted::FittedBernoulliRBM, hidden::AbstractMatrix)
    size(hidden, 2) == size(fitted.weights, 2) || throw(SchemaMismatchError(
        "BernoulliRBM hidden input has $(size(hidden, 2)) components; expected $(size(fitted.weights, 2))."))
    all(value -> isfinite(value) && 0 <= value <= 1, hidden) || throw(UnsupportedDataError(
        "BernoulliRBM hidden values must be finite and lie in [0, 1]."))
    Kernels.sigmoid(hidden * transpose(fitted.weights) .+ transpose(fitted.visible_bias))
end

report(fitted::FittedBernoulliRBM) = fitted.report
