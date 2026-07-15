"""Multinomial naive Bayes for nonnegative count or frequency features."""
struct MultinomialNaiveBayes <: AbstractPredictor
    alpha::Float64
    fit_prior::Bool
    function MultinomialNaiveBayes(; alpha::Real=1.0, fit_prior::Bool=true)
        isfinite(alpha) && alpha > 0 || throw(InvalidHyperparameterError(
            "MultinomialNaiveBayes alpha must be finite and positive."))
        new(Float64(alpha), fit_prior)
    end
end

struct FittedMultinomialNaiveBayes{M,T,L,R,S} <: AbstractFittedEstimator
    model::M
    feature_log_probabilities::Matrix{T}
    class_log_priors::Vector{T}
    classes::Vector{L}
    report::R
    schema::S
end

capabilities(::Type{<:MultinomialNaiveBayes}) = (
    task=:classification, sparse=true, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function fit(model::MultinomialNaiveBayes, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "MultinomialNaiveBayes fitting")
    _validate_numeric_matrix(X, "MultinomialNaiveBayes")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "MultinomialNaiveBayes target has length $(length(y)); expected $(size(X, 1))."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "MultinomialNaiveBayes requires observations and features."))
    all(>=(zero(eltype(X))), X) || throw(UnsupportedDataError(
        "MultinomialNaiveBayes requires nonnegative features."))
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
    length(observation_weights) == length(y) || throw(SchemaMismatchError(
        "MultinomialNaiveBayes weights must match the observation count."))
    all(value -> isfinite(value) && value >= 0, observation_weights) ||
        throw(UnsupportedDataError(
            "MultinomialNaiveBayes weights must be finite and nonnegative."))
    classes = _classification_classes(y)
    class_counts = T[sum(observation_weights[y .== class]) for class in classes]
    all(>(zero(T)), class_counts) || throw(UnsupportedDataError(
        "MultinomialNaiveBayes each class must have positive total weight."))
    feature_counts = Matrix{T}(undef, length(classes), size(X, 2))
    for (class_index, class) in enumerate(classes)
        members = findall(==(class), y)
        feature_counts[class_index, :] .=
            transpose(T.(X[members, :])) * observation_weights[members]
    end
    smoothed = feature_counts .+ T(model.alpha)
    feature_log_probabilities = log.(smoothed ./ sum(smoothed; dims=2))
    class_probabilities = model.fit_prior ? class_counts ./ sum(class_counts) :
        fill(inv(T(length(classes))), length(classes))
    class_log_priors = log.(class_probabilities)
    details = (class_order=copy(classes), class_counts=class_counts,
               alpha=model.alpha, fit_prior=model.fit_prior)
    FittedMultinomialNaiveBayes(model, feature_log_probabilities,
        class_log_priors, classes,
        FitReport(observations=size(X, 1), features=size(X, 2),
                  details=details, context=context),
        with_class_target(infer_schema(X), classes))
end

function _multinomial_joint_log_likelihood(fitted::FittedMultinomialNaiveBayes,
                                           X::AbstractMatrix)
    Matrix(X * transpose(fitted.feature_log_probabilities)) .+
        transpose(fitted.class_log_priors)
end

function predict_proba(fitted::FittedMultinomialNaiveBayes, X::AbstractMatrix)
    _validate_numeric_matrix(X, "MultinomialNaiveBayes")
    _validate_feature_count(fitted.schema, X, "MultinomialNaiveBayes")
    all(>=(zero(eltype(X))), X) || throw(UnsupportedDataError(
        "MultinomialNaiveBayes requires nonnegative features."))
    Kernels.softmax(_multinomial_joint_log_likelihood(fitted, X); dims=2)
end

function predict(fitted::FittedMultinomialNaiveBayes, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))]
     for row in axes(probabilities, 1)]
end

report(fitted::FittedMultinomialNaiveBayes) = fitted.report
