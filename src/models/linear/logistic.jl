"""
L2-regularized logistic regression trained with analytic damped Newton steps.

For more than two classes, Tilia fits one binary problem per sorted class and
normalizes the one-vs-rest scores. The minimized binary objective is the
weighted sum of logistic losses plus `lambda/2 * ||coefficients||²`; the
intercept is never penalized.
"""
struct LogisticRegression{T<:Real} <: AbstractPredictor
    lambda::T
    fit_intercept::Bool
    max_iterations::Int
    tolerance::T
    function LogisticRegression(; lambda::Real=1.0, fit_intercept::Bool=true,
                                max_iterations::Integer=100,
                                tolerance::Real=1e-8)
        isfinite(lambda) && lambda >= 0 || throw(InvalidHyperparameterError(
            "LogisticRegression lambda must be finite and nonnegative; received $lambda."))
        max_iterations > 0 || throw(InvalidHyperparameterError(
            "LogisticRegression max_iterations must be positive; received $max_iterations."))
        isfinite(tolerance) && tolerance > 0 || throw(InvalidHyperparameterError(
            "LogisticRegression tolerance must be finite and positive; received $tolerance."))
        T = promote_type(typeof(lambda), typeof(tolerance))
        new{T}(T(lambda), fit_intercept, Int(max_iterations), T(tolerance))
    end
end

struct FittedLogisticRegression{M,C,I,L,R,S} <: AbstractFittedEstimator
    model::M
    coefficients::C
    intercept::I
    classes::L
    report::R
    schema::S
end

capabilities(::Type{<:LogisticRegression}) = (
    task=:classification, sparse=false, missing=false, weights=true,
    partial_fit=false, probabilistic=true,
)

function _classification_classes(y)
    any(ismissing, y) && throw(UnsupportedDataError(
        "LogisticRegression target cannot contain missing values."))
    classes = try
        sort!(unique(y))
    catch
        throw(UnsupportedDataError(
            "LogisticRegression class labels must have a deterministic sortable ordering."))
    end
    length(classes) >= 2 || throw(UnsupportedDataError(
        "LogisticRegression requires at least two target classes."))
    classes
end

function fit(model::LogisticRegression, X::AbstractMatrix, y::AbstractVector;
             weights=nothing, context=default_context())
    require_cpu(context, "LogisticRegression fitting")
    X isa SparseMatrixCSC && throw(UnsupportedDataError(
        "LogisticRegression sparse fitting is not supported yet; provide a dense matrix."))
    _validate_numeric_matrix(X, "LogisticRegression")
    size(X, 1) == length(y) || throw(SchemaMismatchError(
        "LogisticRegression target has length $(length(y)); expected $(size(X, 1)) observations."))
    size(X, 1) > 0 && size(X, 2) > 0 || throw(UnsupportedDataError(
        "LogisticRegression requires at least one observation and feature."))
    classes = _classification_classes(y)
    T = float(promote_type(eltype(X), weights === nothing ? eltype(X) : eltype(weights)))
    design_features = Matrix{T}(X)
    design = model.fit_intercept ? hcat(design_features, ones(T, size(X, 1))) : design_features
    observation_weights = weights === nothing ? ones(T, length(y)) : T.(weights)
    length(observation_weights) == length(y) || throw(SchemaMismatchError(
        "LogisticRegression weights have length $(length(observation_weights)); expected $(length(y))."))
    all(weight -> isfinite(weight) && weight >= 0, observation_weights) && sum(observation_weights) > 0 ||
        throw(UnsupportedDataError("LogisticRegression weights must be finite, nonnegative, and have a positive sum."))
    penalty_mask = ones(T, size(design, 2))
    model.fit_intercept && (penalty_mask[end] = zero(T))
    trained_classes = length(classes) == 2 ? classes[end:end] : classes
    coefficients = Matrix{T}(undef, size(X, 2), length(trained_classes))
    intercepts = Vector{T}(undef, length(trained_classes))
    traces = Vector{Vector{T}}(undef, length(trained_classes))
    iterations = Vector{Int}(undef, length(trained_classes))
    converged = BitVector(undef, length(trained_classes))
    gradient_norms = Vector{T}(undef, length(trained_classes))
    for (column, class) in enumerate(trained_classes)
        binary_target = T.(y .== class)
        result = try
            Solvers.binary_logistic_newton(design, binary_target;
                weights=observation_weights, lambda=T(model.lambda), penalty_mask=penalty_mask,
                max_iterations=model.max_iterations, tolerance=T(model.tolerance))
        catch error
            error isa SingularException || rethrow()
            throw(NumericalFailureError(
                "LogisticRegression Newton system was singular; increase lambda or remove collinear features."))
        end
        coefficients[:, column] = view(result.parameters, 1:size(X, 2))
        intercepts[column] = model.fit_intercept ? result.parameters[end] : zero(T)
        traces[column] = result.objective_history
        iterations[column] = result.iterations
        converged[column] = result.converged
        gradient_norms[column] = result.gradient_norm
    end
    warnings = all(converged) ? String[] :
        ["One or more class objectives reached max_iterations before convergence."]
    details = (solver=:newton, convergence=converged, iterations=iterations,
               objective_history=traces, gradient_norms=gradient_norms,
               regularization=model.lambda, class_order=copy(classes), strategy=:one_vs_rest)
    fit_report = FitReport(status=all(converged) ? :success : :max_iterations,
        observations=size(X, 1), features=size(X, 2), backend=:cpu,
        warnings=warnings, details=details)
    schema = infer_schema(X)
    schema = Schema(schema.columns; class_order=Any[classes...])
    FittedLogisticRegression(model, coefficients, intercepts, classes, fit_report, schema)
end

function _logistic_scores(fitted::FittedLogisticRegression, X::AbstractMatrix)
    _validate_numeric_matrix(X, "LogisticRegression")
    _validate_feature_count(fitted.schema, X, "LogisticRegression")
    X * fitted.coefficients .+ transpose(fitted.intercept)
end

function predict_proba(fitted::FittedLogisticRegression, X::AbstractMatrix)
    scores = _logistic_scores(fitted, X)
    positive = Kernels.sigmoid(scores)
    if length(fitted.classes) == 2
        return hcat(one(eltype(positive)) .- vec(positive), vec(positive))
    end
    row_sums = sum(positive; dims=2)
    positive ./ row_sums
end

function predict(fitted::FittedLogisticRegression, X::AbstractMatrix)
    probabilities = predict_proba(fitted, X)
    [fitted.classes[argmax(view(probabilities, row, :))] for row in axes(probabilities, 1)]
end

report(fitted::FittedLogisticRegression) = fitted.report
