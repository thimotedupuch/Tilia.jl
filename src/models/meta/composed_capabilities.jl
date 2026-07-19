# A wrapper supports observation weights only when every estimator that it
# forwards those weights to supports them too. Type-level declarations remain
# useful catalog defaults; instance-level declarations capture composition.
_meta_weight_capabilities(model, supported) =
    merge(capabilities(typeof(model)), (weights=supported,))

capabilities(model::OneVsRestClassifier) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::OneVsOneClassifier) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::MultiOutputRegressor) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::MultiOutputClassifier) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::ClassifierChain) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::BaggingRegressor) = _meta_weight_capabilities(
    model, capabilities(model.base_estimator).weights)
capabilities(model::BaggingClassifier) = _meta_weight_capabilities(
    model, capabilities(model.base_estimator).weights)
capabilities(model::VotingRegressor) = _meta_weight_capabilities(
    model, all(estimator -> capabilities(estimator).weights, model.estimators))
capabilities(model::VotingClassifier) = _meta_weight_capabilities(
    model, all(estimator -> capabilities(estimator).weights, model.estimators))
capabilities(model::StackingRegressor) = _meta_weight_capabilities(
    model,
    capabilities(model.final_estimator).weights &&
        all(estimator -> capabilities(estimator).weights, model.estimators),
)
capabilities(model::StackingClassifier) = _meta_weight_capabilities(
    model,
    capabilities(model.final_estimator).weights &&
        all(estimator -> capabilities(estimator).weights, model.estimators),
)
capabilities(model::TransformedTargetRegressor) = _meta_weight_capabilities(
    model, capabilities(model.regressor).weights)
capabilities(model::CalibratedClassifier) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
capabilities(model::ThresholdSelectionWrapper) = _meta_weight_capabilities(
    model, capabilities(model.estimator).weights)
