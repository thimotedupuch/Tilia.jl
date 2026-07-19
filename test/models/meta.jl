using Test
using Statistics
using Tilia

@testset "Meta-estimators and Multi-output Learning" begin
    # Helper binary data
    X_bin = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0; 5.0 4.0; 6.0 2.0]
    y_bin = [1, 1, 2, 2, 2, 1]

    # Helper multiclass data
    X_mc = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0; 5.0 4.0; 6.0 2.0; 7.0 1.0; 8.0 2.0; 9.0 3.0]
    y_mc = [1, 1, 2, 2, 3, 3, 1, 2, 3]

    # Helper regression data
    X_reg = [1.0 2.0; 2.0 0.0; 3.0 1.0; 4.0 3.0; 5.0 4.0; 6.0 2.0]
    y_reg = [2.0, 1.0, 3.0, 7.0, 8.0, 6.0]

    # 1. OneVsRestClassifier
    ovr = fit(OneVsRestClassifier(LogisticRegression(lambda=0.1)), X_mc, y_mc)
    @test length(predict(ovr, X_mc)) == 9
    probs_ovr = predict_proba(ovr, X_mc)
    @test size(probs_ovr) == (9, 3)
    @test all(sum(probs_ovr; dims=2) ≈ ones(9))

    # 2. OneVsOneClassifier
    ovo = fit(OneVsOneClassifier(LogisticRegression(lambda=0.1)), X_mc, y_mc)
    @test length(predict(ovo, X_mc)) == 9
    probs_ovo = predict_proba(ovo, X_mc)
    @test size(probs_ovo) == (9, 3)
    @test all(sum(probs_ovo; dims=2) ≈ ones(9))

    # 3. MultiOutputRegressor & MultiOutputClassifier
    y_multi_reg = [2.0 3.0; 1.0 2.0; 3.0 4.0; 7.0 8.0; 8.0 9.0; 6.0 7.0]
    mor = fit(MultiOutputRegressor(LinearRegression()), X_reg, y_multi_reg)
    preds_mor = predict(mor, X_reg)
    @test size(preds_mor) == (6, 2)

    y_multi_clf = [1 2; 1 1; 2 2; 2 1; 2 2; 1 1]
    moc = fit(MultiOutputClassifier(LogisticRegression(lambda=0.1)), X_bin, y_multi_clf)
    preds_moc = predict(moc, X_bin)
    @test size(preds_moc) == (6, 2)
    probs_moc = predict_proba(moc, X_bin)
    @test length(probs_moc) == 2
    @test size(probs_moc[1]) == (6, 2)

    # 4. ClassifierChain
    chain = fit(ClassifierChain(LogisticRegression(lambda=0.1)), X_bin, y_multi_clf)
    @test size(predict(chain, X_bin)) == (6, 2)
    probs_chain = predict_proba(chain, X_bin)
    @test length(probs_chain) == 2

    # 5. BaggingClassifier & BaggingRegressor
    bag_clf = fit(BaggingClassifier(LogisticRegression(lambda=0.1), n_estimators=5, max_samples=0.8), X_bin, y_bin)
    @test length(predict(bag_clf, X_bin)) == 6
    @test size(predict_proba(bag_clf, X_bin)) == (6, 2)

    bag_reg = fit(BaggingRegressor(LinearRegression(), n_estimators=5, max_samples=0.8), X_reg, y_reg)
    @test length(predict(bag_reg, X_reg)) == 6

    # 6. VotingClassifier & VotingRegressor
    vote_clf_soft = fit(VotingClassifier(LogisticRegression(lambda=0.1), LogisticRegression(lambda=0.2); voting=:soft), X_bin, y_bin)
    @test length(predict(vote_clf_soft, X_bin)) == 6
    @test size(predict_proba(vote_clf_soft, X_bin)) == (6, 2)

    vote_clf_hard = fit(VotingClassifier(LogisticRegression(lambda=0.1), LogisticRegression(lambda=0.2); voting=:hard), X_bin, y_bin)
    @test length(predict(vote_clf_hard, X_bin)) == 6

    vote_reg = fit(VotingRegressor(LinearRegression(), RidgeRegression(); weights=[1.0, 2.0]), X_reg, y_reg)
    @test length(predict(vote_reg, X_reg)) == 6

    # 7. StackingClassifier & StackingRegressor
    stack_clf = fit(StackingClassifier((LogisticRegression(lambda=0.1), LogisticRegression(lambda=0.2)), LogisticRegression(lambda=0.1); cv=KFold(3)), X_bin, y_bin)
    @test length(predict(stack_clf, X_bin)) == 6
    @test size(predict_proba(stack_clf, X_bin)) == (6, 2)

    stack_reg = fit(StackingRegressor((LinearRegression(), RidgeRegression()), RidgeRegression(); cv=KFold(3)), X_reg, y_reg)
    @test length(predict(stack_reg, X_reg)) == 6

    # 8. TransformedTargetRegressor
    ttr = fit(TransformedTargetRegressor(LinearRegression(), func=log, inverse_func=exp), X_reg, y_reg)
    @test length(predict(ttr, X_reg)) == 6

    # 9. CalibratedClassifier
    cal = fit(CalibratedClassifier(LogisticRegression(lambda=0.1); cv=KFold(3)), X_bin, y_bin)
    @test length(predict(cal, X_bin)) == 6
    @test size(predict_proba(cal, X_bin)) == (6, 2)

    # 10. ThresholdSelectionWrapper
    tsw = fit(ThresholdSelectionWrapper(LogisticRegression(lambda=0.1); metric=:f1, cv=KFold(3)), X_bin, y_bin)
    @test length(predict(tsw, X_bin)) == 6
    @test size(predict_proba(tsw, X_bin)) == (6, 2)

    # Weight support is the intersection of the wrapped estimators' support.
    weighted_models = (
        OneVsRestClassifier(LogisticRegression()),
        MultiOutputRegressor(RidgeRegression()),
        VotingClassifier(LogisticRegression(), LogisticRegression()),
        StackingRegressor((RidgeRegression(), LinearRegression()), RidgeRegression()),
        TransformedTargetRegressor(RidgeRegression()),
    )
    @test all(model -> capabilities(model).weights, weighted_models)

    unweighted_models = (
        OneVsRestClassifier(KNeighborsClassifier()),
        OneVsOneClassifier(KNeighborsClassifier()),
        MultiOutputClassifier(KNeighborsClassifier()),
        MultiOutputRegressor(KNeighborsRegressor()),
        ClassifierChain(KNeighborsClassifier()),
        BaggingClassifier(KNeighborsClassifier()),
        BaggingRegressor(KNeighborsRegressor()),
        VotingClassifier(LogisticRegression(), KNeighborsClassifier()),
        VotingRegressor(RidgeRegression(), KNeighborsRegressor()),
        StackingClassifier((LogisticRegression(), KNeighborsClassifier()), LogisticRegression()),
        StackingRegressor((RidgeRegression(),), KNeighborsRegressor()),
        TransformedTargetRegressor(KNeighborsRegressor()),
        CalibratedClassifier(KNeighborsClassifier()),
        ThresholdSelectionWrapper(KNeighborsClassifier()),
    )
    @test all(model -> !capabilities(model).weights, unweighted_models)
end
