"""
Fit a fresh estimator (including all pipeline transforms) inside each fold and
return scores, fitted folds, reports, and the exact non-overlapping indices.
"""
function cross_validate(model::AbstractEstimator, input, y::AbstractVector;
                        cv::KFold=KFold(), scoring=nothing, context=default_context())
    X = input isa AbstractMatrix || input isa ColumnTable ? input : column_table(input)
    size(X, 1) == length(y) || throw(SchemaMismatchError("features and target observation counts must agree."))
    scoring === nothing && (scoring = capabilities(model).task === :classification ? accuracy_score :
                                      root_mean_squared_error)
    folds = split(cv, length(y))
    fitted_models = AbstractFittedEstimator[]
    fold_reports = FitReport[]
    scores = Float64[]
    train_indices = Vector{Int}[]
    test_indices = Vector{Int}[]
    for (fold, (train, test)) in enumerate(folds)
        fold_context = derive_context(context, :cross_validation, :fold, fold)
        fitted = fit(model, select_rows(X, train), y[train]; context=fold_context)
        predictions = predict(fitted, select_rows(X, test))
        push!(fitted_models, fitted)
        push!(fold_reports, report(fitted))
        push!(scores, scoring(y[test], predictions))
        push!(train_indices, train)
        push!(test_indices, test)
    end
    CrossValidationResult(scores, fitted_models, fold_reports, train_indices, test_indices)
end

"""Evaluate an estimator with leakage-safe cross-validation."""
evaluate(model::AbstractEstimator, input, y::AbstractVector; kwargs...) =
    cross_validate(model, input, y; kwargs...)

function _parameter_combinations(parameter_grid::NamedTuple)
    names = keys(parameter_grid)
    isempty(names) && throw(InvalidHyperparameterError("parameter_grid cannot be empty."))
    grid_values = Base.values(parameter_grid)
    all(value -> value isa AbstractVector || value isa Tuple, grid_values) ||
        throw(InvalidHyperparameterError("Every parameter-grid value must be a vector or tuple."))
    [NamedTuple{names}(combination) for combination in Iterators.product(grid_values...)]
end

function _replace_parameters(model::AbstractEstimator, parameters::NamedTuple)
    valid_names = propertynames(model)
    all(name -> name in valid_names, keys(parameters)) || throw(InvalidHyperparameterError(
        "Parameter grid contains fields not present in $(nameof(typeof(model)))."))
    merged = (; (name => (haskey(parameters, name) ? parameters[name] : getproperty(model, name))
                  for name in valid_names)...)
    try
        Base.typename(typeof(model)).wrapper(; merged...)
    catch error
        error isa TiliaError && rethrow()
        throw(InvalidHyperparameterError(
            "Could not construct $(nameof(typeof(model))) from tuned parameters: $(sprint(showerror, error))."))
    end
end

function _copied_context(context)
    FitContext(backend=context.backend, rng=copy(context.rng), numerics=context.numerics,
               deterministic=context.deterministic, cache=context.cache,
               root_seed=context.root_seed, stream_id=context.stream_id)
end

"""
Exhaustively evaluate a named parameter grid with cross-validation.

Classification scores are maximized by default and regression scores are
minimized. Set `maximize` explicitly for custom scoring functions.
"""
function tune(model::AbstractEstimator, input, y::AbstractVector;
              parameter_grid::NamedTuple, cv::KFold=KFold(), scoring=nothing,
              maximize=nothing, refit::Bool=true, context=default_context())
    combinations = _parameter_combinations(parameter_grid)
    isempty(combinations) && throw(InvalidHyperparameterError("parameter_grid cannot be empty."))
    maximize === nothing && (maximize = capabilities(model).task === :classification)
    trials = NamedTuple[]
    best_score = maximize ? -Inf : Inf
    best_model = nothing
    best_parameters = nothing
    for (trial, parameters) in enumerate(combinations)
        candidate = _replace_parameters(model, parameters)
        trial_context = derive_context(context, :tuning, :trial, trial)
        evaluation = cross_validate(candidate, input, y; cv=cv, scoring=scoring,
                                    context=trial_context)
        score = mean(evaluation.scores)
        push!(trials, (parameters=parameters, score=score,
                       fold_scores=copy(evaluation.scores), reports=evaluation.fold_reports))
        better = maximize ? score > best_score : score < best_score
        if better
            best_score, best_model, best_parameters = score, candidate, parameters
        end
    end
    fitted = refit ? fit(best_model, input, y;
                         context=derive_context(context, :tuning, :refit)) : nothing
    TuningResult(best_model, best_parameters, best_score, trials, fitted)
end
