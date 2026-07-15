@testset "Classification diagnostics scikit-learn reference fixture" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "classification_curves_sklearn.toml"))
    @test fixture["source"]["version"] == "1.9.0"
    targets = fixture["input"]["targets"]
    scores = fixture["input"]["scores"]
    atol = fixture["tolerance"]["absolute"]
    rtol = fixture["tolerance"]["relative"]

    roc = roc_curve(targets, scores)
    @test roc.false_positive_rate ≈ fixture["roc"]["false_positive_rate"] atol=atol rtol=rtol
    @test roc.true_positive_rate ≈ fixture["roc"]["true_positive_rate"] atol=atol rtol=rtol
    @test roc.thresholds == fixture["roc"]["thresholds"]
    @test area_under_curve(roc) ≈ fixture["roc"]["area"] atol=atol rtol=rtol

    precision_recall = precision_recall_curve(targets, scores)
    @test precision_recall.precision ≈
          reverse(fixture["precision_recall"]["precision"]) atol=atol rtol=rtol
    @test precision_recall.recall ≈
          reverse(fixture["precision_recall"]["recall"]) atol=atol rtol=rtol
    @test precision_recall.thresholds ==
          [Inf; reverse(fixture["precision_recall"]["thresholds"])]

    calibration = calibration_curve(targets, scores; n_bins=2)
    @test calibration.fraction_positive ≈
          fixture["calibration"]["fraction_positive"] atol=atol rtol=rtol
    @test calibration.mean_predicted_probability ≈
          fixture["calibration"]["mean_predicted_probability"] atol=atol rtol=rtol
end
