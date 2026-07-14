abstract type AbstractGaussianClassifier <: AbstractPredictor end

"""Gaussian naive Bayes with independently estimated per-class feature variances."""
struct GaussianNaiveBayes <: AbstractGaussianClassifier
    var_smoothing::Float64
    function GaussianNaiveBayes(; var_smoothing::Real=1e-9)
        isfinite(var_smoothing) && var_smoothing >= 0 || throw(InvalidHyperparameterError(
            "GaussianNaiveBayes var_smoothing must be finite and nonnegative."))
        new(Float64(var_smoothing))
    end
end

"""Gaussian discriminant analysis using one regularized covariance shared by all classes."""
struct LinearDiscriminantAnalysis <: AbstractGaussianClassifier
    regularization::Float64
    function LinearDiscriminantAnalysis(; regularization::Real=1e-6)
        isfinite(regularization) && regularization >= 0 || throw(InvalidHyperparameterError(
            "LinearDiscriminantAnalysis regularization must be finite and nonnegative."))
        new(Float64(regularization))
    end
end

"""Gaussian discriminant analysis using a regularized covariance for each class."""
struct QuadraticDiscriminantAnalysis <: AbstractGaussianClassifier
    regularization::Float64
    function QuadraticDiscriminantAnalysis(; regularization::Real=1e-6)
        isfinite(regularization) && regularization >= 0 || throw(InvalidHyperparameterError(
            "QuadraticDiscriminantAnalysis regularization must be finite and nonnegative."))
        new(Float64(regularization))
    end
end


struct FittedGaussianNaiveBayes{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    means::Matrix{T}
    variances::Matrix{T}
    priors::Vector{T}
    classes::Vector{L}
    report::R
    schema::S
end

struct FittedDiscriminantAnalysis{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    means::Matrix{T}
    precisions::Vector{Matrix{T}}
    log_determinants::Vector{T}
    priors::Vector{T}
    classes::Vector{L}
    report::R
    schema::S
end

capabilities(::Type{<:AbstractGaussianClassifier}) = (task=:classification, sparse=false,
    missing=false, weights=true, partial_fit=false, probabilistic=true)

function _prepare_gaussian_data(model, X, y, weights, context)
    name = string(nameof(typeof(model)))
    require_cpu(context, "$name fitting")
    _validate_numeric_matrix(X, name)
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "$name target has length $(length(y)); expected $(size(X, 1)) observations."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "$name requires at least one observation and feature."))
    classes = _classification_classes(y)
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    data = Matrix{T}(X)
    observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
    length(observation_weights) == length(y) || throw(SchemaMismatchError(
        "$name weights have length $(length(observation_weights)); expected $(length(y))."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) ||
        throw(UnsupportedDataError("$name weights must be finite and nonnegative."))
    class_weights = T[sum(observation_weights[y .== class]) for class in classes]
    all(>(zero(T)), class_weights) || throw(UnsupportedDataError(
        "$name each class must have positive total weight."))
    data, observation_weights, classes, class_weights
end

function fit(model::GaussianNaiveBayes, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    data, observation_weights, classes, class_weights =
        _prepare_gaussian_data(model, X, y, weights, context)
    T = eltype(data)
    means = Matrix{T}(undef, length(classes), size(X, 2))
    variances = similar(means)
    for (index, class) in enumerate(classes)
        members = findall(==(class), y)
        local_weights = observation_weights[members]
        total = class_weights[index]
        class_data = view(data, members, :)
        means[index, :] .= vec(sum(class_data .* local_weights; dims=1)) ./ total
        centered = class_data .- view(means, index:index, :)
        variances[index, :] .= vec(sum(abs2.(centered) .* local_weights; dims=1)) ./ total
    end
    global_variance = vec(var(data; dims=1, corrected=false))
    epsilon = T(model.var_smoothing) * maximum(global_variance; init=zero(T))
    variances .+= max(epsilon, eps(T))
    priors = class_weights ./ sum(class_weights)
    details = (class_order=copy(classes), class_counts=class_weights,
               variance_smoothing=epsilon)
    schema = infer_schema(X)
    schema = Schema(schema.columns; class_order=Any[classes...])
    FittedGaussianNaiveBayes(model, means, variances, priors, classes,
        FitReport(observations=size(X, 1), features=size(X, 2), backend=:cpu, details=details), schema)
end

function _regularized_precision(covariance, regularization, model_name)
    T = eltype(covariance)
    regularized = Hermitian(covariance + T(regularization) * I)
    factor = try
        cholesky(regularized)
    catch error
        error isa PosDefException || rethrow()
        throw(NumericalFailureError(
            "$model_name covariance is not positive definite; increase regularization."))
    end
    Matrix(inv(factor)), T(2sum(log, diag(factor.L)))
end

function fit(model::Union{LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis},
             X::AbstractMatrix, y::AbstractVector; weights=nothing, context=default_context())
    data, observation_weights, classes, class_weights =
        _prepare_gaussian_data(model, X, y, weights, context)
    T = eltype(data)
    means = Matrix{T}(undef, length(classes), size(X, 2))
    scatters = Vector{Matrix{T}}(undef, length(classes))
    for (index, class) in enumerate(classes)
        members = findall(==(class), y)
        local_weights = observation_weights[members]
        class_data = view(data, members, :)
        means[index, :] .= vec(sum(class_data .* local_weights; dims=1)) ./ class_weights[index]
        centered = class_data .- view(means, index:index, :)
        scatters[index] = transpose(centered) * (centered .* local_weights)
    end
    covariances = if model isa LinearDiscriminantAnalysis
        shared = sum(scatters) ./ sum(class_weights)
        fill(shared, length(classes))
    else
        [scatters[index] ./ class_weights[index] for index in eachindex(classes)]
    end
    precisions = Vector{Matrix{T}}(undef, length(classes))
    log_determinants = Vector{T}(undef, length(classes))
    for index in eachindex(classes)
        precisions[index], log_determinants[index] = _regularized_precision(
            covariances[index], model.regularization, string(nameof(typeof(model))))
    end
    priors = class_weights ./ sum(class_weights)
    details = (class_order=copy(classes), class_counts=class_weights,
               covariance=model isa LinearDiscriminantAnalysis ? :shared : :class_specific,
               regularization=model.regularization)
    schema = infer_schema(X)
    schema = Schema(schema.columns; class_order=Any[classes...])
    FittedDiscriminantAnalysis(model, means, precisions, log_determinants, priors,
        classes, FitReport(observations=size(X, 1), features=size(X, 2),
        backend=:cpu, details=details), schema)
end

function _joint_log_likelihood(fitted::FittedGaussianNaiveBayes, X)
    T = eltype(fitted.means)
    scores = Matrix{T}(undef, size(X, 1), length(fitted.classes))
    constant_term = T(size(X, 2) * log(2pi))
    for class in eachindex(fitted.classes)
        centered = X .- view(fitted.means, class:class, :)
        scores[:, class] .= log(fitted.priors[class]) .-
            T(0.5) .* (constant_term + sum(log, view(fitted.variances, class, :)) .+
            vec(sum(abs2.(centered) ./ view(fitted.variances, class:class, :); dims=2)))
    end
    scores
end

function _joint_log_likelihood(fitted::FittedDiscriminantAnalysis, X)
    T = eltype(fitted.means)
    scores = Matrix{T}(undef, size(X, 1), length(fitted.classes))
    constant_term = T(size(X, 2) * log(2pi))
    for class in eachindex(fitted.classes)
        centered = X .- view(fitted.means, class:class, :)
        quadratic = vec(sum((centered * fitted.precisions[class]) .* centered; dims=2))
        scores[:, class] .= log(fitted.priors[class]) .-
            T(0.5) .* (constant_term + fitted.log_determinants[class] .+ quadratic)
    end
    scores
end

function predict_proba(fitted::Union{FittedGaussianNaiveBayes,FittedDiscriminantAnalysis},
                       X::AbstractMatrix)
    name = string(nameof(typeof(fitted.model)))
    _validate_numeric_matrix(X, name)
    _validate_feature_count(fitted.schema, X, name)
    Kernels.softmax(_joint_log_likelihood(fitted, Matrix{eltype(fitted.means)}(X)); dims=2)
end

function predict(fitted::Union{FittedGaussianNaiveBayes,FittedDiscriminantAnalysis},
                 X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(probabilities, 1)]
end

report(fitted::Union{FittedGaussianNaiveBayes,FittedDiscriminantAnalysis}) = fitted.report
