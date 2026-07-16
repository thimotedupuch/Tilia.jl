using Test
using Tilia
using TiliaMakieRecipes
using TiliaMakieRecipes.Makie

@testset "Tilia diagnostic plots" begin
    confusion = ConfusionMatrix([4 1; 2 5], [:negative, :positive])
    confusion_plot = Makie.plot(confusion)
    @test confusion_plot isa Makie.FigureAxisPlot
    @test confusion_plot.axis.xlabel[] == "Predicted class"
    @test confusion_plot.axis.ylabel[] == "True class"
    @test confusion_plot.axis.xticks[] == ([1, 2], ["negative", "positive"])
    @test length(confusion_plot.plot.plots) == 2
    @test confusion_plot.plot.plots[1] isa Makie.Heatmap
    @test confusion_plot.plot.plots[2] isa Makie.Text

    without_values = Makie.plot(confusion; show_values = false)
    @test length(without_values.plot.plots) == 1

    roc = ROCResult([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], [Inf, 0.5, -Inf])
    roc_plot = Makie.plot(roc)
    @test roc_plot.axis.xlabel[] == "False-positive rate"
    @test roc_plot.axis.ylabel[] == "True-positive rate"
    @test length(roc_plot.plot.plots) == 2
    @test roc_plot.plot.plots[1] isa Makie.ABLines
    @test roc_plot.plot.plots[2] isa Makie.Lines
    @test length(Makie.plot(roc; show_reference = false).plot.plots) == 1

    precision_recall = PrecisionRecallResult([1.0, 0.5], [0.0, 1.0], [Inf, 0.2])
    precision_recall_plot = Makie.plot(precision_recall)
    @test precision_recall_plot.axis.xlabel[] == "Recall"
    @test precision_recall_plot.axis.ylabel[] == "Precision"
    @test only(precision_recall_plot.plot.plots) isa Makie.Lines

    calibration = CalibrationResult([0.2, 0.8], [0.1, 0.9], [1.0, 9.0],
                                    [0.0, 0.5, 1.0])
    calibration_plot = Makie.plot(calibration)
    @test calibration_plot.axis.title[] == "Calibration curve"
    @test length(calibration_plot.plot.plots) == 2
    @test calibration_plot.plot.plots[1] isa Makie.ABLines
    @test calibration_plot.plot.plots[2] isa Makie.ScatterLines

    importance = PermutationImportanceResult(
        0.9, [0.2 0.3; 0.0 0.1], [0.25, 0.05], [0.05, 0.05], [:x1, :x2])
    importance_plot = Makie.plot(importance)
    @test importance_plot.axis.xticks[] == ([1, 2], ["x1", "x2"])
    @test length(importance_plot.plot.plots) == 3
    @test importance_plot.plot.plots[1] isa Makie.BarPlot
    @test importance_plot.plot.plots[2] isa Makie.Errorbars
    @test importance_plot.plot.plots[3] isa Makie.HLines

    cross_validation = CrossValidationResult(
        [0.8, 0.9, 0.85], Any[], Any[], [Int[] for _ in 1:3], [Int[] for _ in 1:3])
    cross_validation_plot = Makie.plot(cross_validation)
    @test cross_validation_plot.axis.xlabel[] == "Fold"
    @test cross_validation_plot.axis.xticks[] == [1, 2, 3]
    @test length(cross_validation_plot.plot.plots) == 3
    @test cross_validation_plot.plot.plots[1] isa Makie.ScatterLines
    @test cross_validation_plot.plot.plots[2] isa Makie.Band
    @test cross_validation_plot.plot.plots[3] isa Makie.HLines

    optimization = OptimizationTrace([3.0, 2.0, 1.5], true)
    optimization_plot = Makie.plot(optimization)
    @test optimization_plot.axis.xlabel[] == "Iteration"
    @test optimization_plot.axis.subtitle[] == "Converged"
    @test only(optimization_plot.plot.plots) isa Makie.ScatterLines

    overridden = Makie.plot(optimization; axis = (ylabel = "Loss",))
    @test overridden.axis.xlabel[] == "Iteration"
    @test overridden.axis.ylabel[] == "Loss"
end

@testset "Model diagnostic plots" begin
    X = [-2.0 -1.8; -1.5 -2.1; -1.8 -1.2; 1.7 1.8; 2.1 1.3; 1.4 2.2]
    y = [:left, :left, :left, :right, :right, :right]
    target = 2 .* X[:, 1] .- X[:, 2] .+ 0.1

    kmeans = fit(KMeans(n_clusters=2, n_init=1), X)
    @test clusterplot(kmeans, X) isa Makie.Figure
    mixture = fit(GaussianMixture(n_components=2, n_init=1), X)
    @test mixturedensityplot(mixture, X) isa Makie.Figure
    hierarchy = fit(AgglomerativeClustering(n_clusters=1), X)
    @test TiliaMakieRecipes.dendrogram(hierarchy) isa Makie.Figure

    tree = fit(DecisionTreeClassifier(max_depth=2), X, y)
    @test decisionboundaryplot(tree, X, y) isa Makie.Figure
    @test treeplot(tree) isa Makie.Figure
    @test learningcurveplot(DecisionTreeClassifier(max_depth=2), X, y;
                            train_sizes=[0.7, 1.0]) isa Makie.Figure
    @test validationcurveplot(DecisionTreeClassifier(), X, y;
                              parameter=:max_depth, values=[1, 2]) isa Makie.Figure

    linear = fit(LinearRegression(), X, target)
    @test residualplot(linear, X, target) isa Makie.Figure
    @test coefficientplot(linear) isa Makie.Figure
    @test regularizationpathplot(Lasso(max_iterations=50), X, target;
                                 lambdas=[0.01, 0.1]) isa Makie.Figure

    neighbors = fit(NearestNeighbors(n_neighbors=2), X)
    @test neighborhoodplot(neighbors, X[1:2, :]; n_neighbors=2) isa Makie.Figure
    forest = fit(IsolationForest(n_estimators=5), X)
    @test anomalyscoreplot(forest, X) isa Makie.Figure

    trials = NamedTuple[(parameters=(depth=1, rate=0.1), score=0.7),
                        (parameters=(depth=2, rate=0.1), score=0.8),
                        (parameters=(depth=1, rate=0.2), score=0.75),
                        (parameters=(depth=2, rate=0.2), score=0.85)]
    tuning = TuningResult(nothing, (depth=2, rate=0.2), 0.85, trials, nothing)
    @test tuningheatmap(tuning; xparameter=:depth, yparameter=:rate) isa Makie.Figure
    @test tuningparallelplot(tuning) isa Makie.Figure
    cv1 = CrossValidationResult([0.7, 0.8, 0.75], Any[], Any[], [Int[] for _ in 1:3], [Int[] for _ in 1:3])
    cv2 = CrossValidationResult([0.8, 0.86, 0.82], Any[], Any[], [Int[] for _ in 1:3], [Int[] for _ in 1:3])
    @test modelcomparisonplot([cv1, cv2]; names=["A", "B"]) isa Makie.Figure
end

@testset "Three-dimensional plots" begin
    X = [-2.0 -1.8 -1.0; -1.5 -2.1 -1.3; -1.8 -1.2 -1.7;
          1.7 1.8 1.2; 2.1 1.3 1.8; 1.4 2.2 1.5]
    y = [:left, :left, :left, :right, :right, :right]
    pca = fit(PCA(n_components=3), X)
    @test projectionplot3d(pca, X; groups=y) isa Makie.Figure
    @test pointcloudplot(X; groups=y, edges=[(1,2), (4,5)]) isa Makie.Figure
    mixture = fit(GaussianMixture(n_components=2, n_init=1), X)
    @test clusterplot3d(mixture, X) isa Makie.Figure
    neighbors = fit(NearestNeighbors(n_neighbors=2), X)
    @test neighborhoodplot3d(neighbors, X[[1,4],:]; n_neighbors=2) isa Makie.Figure

    X2 = X[:,1:2]; target = X2[:,1] .- 0.5X2[:,2]
    linear = fit(LinearRegression(), X2, target)
    @test regressionsurfaceplot(linear, X2, target; resolution=12) isa Makie.Figure
    trials = NamedTuple[(parameters=(depth=1, rate=0.1), score=0.70),
                        (parameters=(depth=2, rate=0.1), score=0.82),
                        (parameters=(depth=1, rate=0.2), score=0.76),
                        (parameters=(depth=2, rate=0.2), score=0.86)]
    tuning = TuningResult(nothing, (depth=2, rate=0.2), 0.86, trials, nothing)
    @test tuninglandscapeplot(tuning; xparameter=:depth, yparameter=:rate) isa Makie.Figure
end

@testset "Explanatory plots" begin
    X = [0.0 0.0; 0.2 0.1; 1.0 0.8; 1.2 1.1; 2.0 0.1; 2.2 0.2]
    target = X[:,1] .+ 0.4X[:,2]
    regressor = fit(LinearRegression(),X,target)
    @test partialdependenceplot(regressor,X;feature=1,resolution=8,observations=3) isa Makie.Figure
    @test partialdependenceplot(regressor,X;feature=(1,2),resolution=6) isa Makie.Figure
    @test partialdependenceplot(regressor,X;feature=(1,2),resolution=6,surface=true) isa Makie.Figure
    y=[:a,:a,:b,:b,:c,:c]
    classifier=fit(DecisionTreeClassifier(max_depth=3),X,y)
    @test partialdependenceplot(classifier,X;feature=1,target=:c,resolution=6) isa Makie.Figure
    @test probabilitysimplexplot(classifier,X;groups=y) isa Makie.Figure
    @test_throws DimensionMismatch probabilitysimplexplot([0.5 0.5; 0.2 0.8])
end

@testset "Dimensionality-reduction plots" begin
    X = [1.0 2.0 0.0 1.0;
         2.0 3.0 1.0 0.0;
         3.0 4.0 0.5 1.0;
         5.0 1.0 3.0 2.0;
         6.0 2.0 4.0 3.0;
         7.0 1.0 3.5 4.0]
    groups = [:a, :a, :a, :b, :b, :b]
    fitted = fit(PCA(n_components=4), X)

    projection = projectionplot(fitted, X; groups=groups)
    @test projection isa Makie.FigureAxisPlot
    @test occursin("PC1", projection.axis.xlabel[])
    @test length(projection.plot.plots) == 5

    scree = screeplot(fitted; threshold=0.9)
    @test scree.axis.xticks[] === Makie.automatic
    @test length(scree.plot.plots) == 4
    @test scree.plot.plots[1] isa Makie.BarPlot

    bi = biplot(fitted, X; groups=groups,
                feature_names=["a", "b", "c", "d"])
    @test bi.axis.title[] == "PCA biplot"
    @test length(bi.plot.plots) == 3
    @test bi.plot.plots[2] isa Makie.Arrows2D

    loadings = loadingsplot(fitted; component=2,
                            feature_names=["a", "b", "c", "d"])
    @test loadings.axis.title[] == "Component 2 loadings"
    @test loadings.axis.xticks[] == ([1, 2, 3, 4],
                                     loadings.plot.data[].names)
    @test loadings.plot.plots[1] isa Makie.BarPlot

    components = componentplot(fitted; shape=(2, 2), columns=2)
    @test components isa Makie.FigureAxisPlot
    @test components.plot.plots[1] isa Makie.Heatmap
    @test length(components.plot.plots) == 5

    reconstruction = reconstructionplot(fitted, X; shape=(2, 2), observations=1:3)
    @test reconstruction isa Makie.FigureAxisPlot
    @test reconstruction.plot.plots[1] isa Makie.Heatmap
    @test length(reconstruction.plot.plots) == 4

    nmf = fit(NMF(n_components=2, max_iterations=20), abs.(X))
    @test componentplot(nmf; shape=(2, 2)).plot.plots[1] isa Makie.Heatmap

    @test_throws DimensionMismatch componentplot(fitted; shape=(3, 2))
    @test_throws ArgumentError screeplot(fitted; threshold=1.1)
end
