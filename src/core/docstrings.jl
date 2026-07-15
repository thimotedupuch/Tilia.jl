@doc """Fit an immutable estimator specification and return fitted state.

```julia
fitted = fit(MeanRegressor(), [1.0;; 2.0], [1.0, 3.0])
```
""" fit

@doc """Predict outputs from fitted state. `predict(fitted, X)` keeps observations in rows.""" predict
@doc """Return class or component probabilities from supported fitted estimators.""" predict_proba
@doc """Apply a fitted transformer to new observations.""" transform
@doc """Map transformed observations back to feature space when supported.""" inverse_transform
@doc """Update estimators that explicitly declare online-learning support; unsupported models raise `UnsupportedDataError`.""" partial_fit
@doc """Return the machine-readable `FitReport` associated with fitted state.""" report
@doc """Describe accepted input semantics for an estimator.""" input_contract
@doc """Describe the output schema rule for an estimator.""" output_schema
@doc """Create a fresh deterministic CPU fit context.""" default_context

@doc """Abstract supertype for feature transformers such as `Standardize`.""" AbstractTransformer
@doc """Abstract supertype for predictors such as `LinearRegression`.""" AbstractPredictor
@doc """Host CPU execution backend. Use `FitContext(backend=CPUBackend())`.""" CPUBackend
@doc """Optional Reactant backend with explicit device and fallback policy.""" ReactantBackend

@doc """Select matrix indices or table column names, for example `Select(:age, :income)`.""" Select
@doc """Fit several transformers to the same input, for example `Parallel(PCA(), Standardize())`.""" Parallel
@doc """Concatenate tuple-valued transformer branches column-wise.""" Concatenate
@doc """Apply transformers to named column groups, for example `ColumnMap(:age => Standardize())`.""" ColumnMap

@doc """Center and/or scale numeric features with population standard deviations.""" Standardize
@doc """Exact brute-force neighbor index. Call `kneighbors(fit(NearestNeighbors(), X), query)`.""" NearestNeighbors

@doc """Numeric CART classifier with Gini or entropy impurity.""" DecisionTreeClassifier
@doc """Numeric CART regressor with squared-error impurity.""" DecisionTreeRegressor
@doc """Bootstrap ensemble of randomized classification trees.""" RandomForestClassifier
@doc """Bootstrap ensemble of randomized regression trees.""" RandomForestRegressor
@doc """Extremely randomized classification-tree ensemble.""" ExtraTreesClassifier
@doc """Extremely randomized regression-tree ensemble.""" ExtraTreesRegressor
@doc """Quantile-binned gradient-boosting classifier.""" HistGradientBoostingClassifier
@doc """Quantile-binned gradient-boosting regressor.""" HistGradientBoostingRegressor
@doc """One-hidden-layer probabilistic classifier trained by batch backpropagation.""" MLPClassifier
@doc """One-hidden-layer regressor trained by batch backpropagation.""" MLPRegressor

@doc """Confusion-matrix values and deterministic label ordering.""" ConfusionMatrix
@doc """False-positive rates, true-positive rates, and thresholds for an ROC curve.""" ROCResult
@doc """Fold scores, fitted folds, reports, and exact split indices.""" CrossValidationResult
@doc """Objective values and convergence status from iterative optimization.""" OptimizationTrace

@doc """Mean squared prediction error, optionally weighted.""" mean_squared_error
@doc """Square root of `mean_squared_error`, optionally weighted.""" root_mean_squared_error

const PUBLIC_EXAMPLE_NAMES = Set{Symbol}()
const PUBLIC_EXAMPLES = Dict{Symbol,String}()
macro _example(binding, code)
    name = binding isa Symbol ? binding : error("example binding must be a symbol")
    push!(PUBLIC_EXAMPLE_NAMES, name)
    PUBLIC_EXAMPLES[name] = String(code)
    :(nothing)
end


@_example fit "fitted = fit(MeanRegressor(), [1.0;; 2.0], [1.0, 3.0])"
@_example predict "predict(fit(MeanRegressor(), [1.0;; 2.0], [1.0, 3.0]), [4.0;;])"
@_example predict_proba "predict_proba(fit(LogisticRegression(), [-1.0;; 1.0], [:a, :b]), [0.0;;])"
@_example transform "transform(fit(Standardize(), [1.0;; 2.0]), [1.5;;])"
@_example inverse_transform "inverse_transform(fit(Standardize(), [1.0;; 2.0]), [0.0;;])"
@_example partial_fit "online = partial_fit(MeanRegressor(), [1.0;;], [2.0])"
@_example evaluate "evaluate(RidgeRegression(), randn(12, 2), randn(12); cv=KFold(3))"
@_example tune "tune(RidgeRegression(), randn(12, 2), randn(12); parameter_grid=(lambda=[0.1, 1.0],))"
@_example report "report(fit(MeanRegressor(), [1.0;;], [2.0]))"
@_example save_model "save_model(\"model\", fit(MeanRegressor(), [1.0;;], [2.0]))"
@_example load_model "fitted = load_model(\"model\")"

@_example Chain "Chain(Standardize(), LogisticRegression())"
@_example Parallel "Parallel(Standardize(), PCA(n_components=1))"
@_example ColumnMap "ColumnMap(:age => Standardize())"
@_example Select "Select(:age, :income)"
@_example Concatenate "Concatenate()"

@_example CPUBackend "context = FitContext(backend=CPUBackend())"
@_example ReactantBackend "backend = ReactantBackend(fallback=:cpu)"
@_example NumericsPolicy "policy = NumericsPolicy(Float32; accumulation_type=Float64)"
@_example FitContext "context = FitContext(seed=42)"
@_example CompilationCache "cache = CompilationCache()"
@_example default_context "context = default_context()"
@_example derive_context "fold_context = derive_context(FitContext(seed=42), :fold, 1)"

@_example ConfusionMatrix "result = confusion_matrix([:a, :b], [:a, :a])"
@_example ROCResult "result = ROCResult([0.0, 1.0], [0.0, 1.0], [Inf, 0.0])"
@_example CrossValidationResult "result = cross_validate(RidgeRegression(), randn(8, 2), randn(8); cv=KFold(2))"
@_example OptimizationTrace "trace = OptimizationTrace([2.0, 1.0], true)"
@_example TuningResult "result = tune(RidgeRegression(), randn(8, 2), randn(8); parameter_grid=(lambda=[0.1],))"

@_example AbstractEstimator "struct MyEstimator <: AbstractEstimator end"
@_example AbstractFittedEstimator "struct MyFitted <: AbstractFittedEstimator end"
@_example AbstractTransformer "struct MyTransform <: AbstractTransformer end"
@_example AbstractPredictor "struct MyPredictor <: AbstractPredictor end"

@_example MeanRegressor "model = MeanRegressor()"
@_example Standardize "transformer = Standardize(center=true, scale=true)"
@_example Dataset "data = Dataset(randn(4, 2); target=randn(4))"
@_example Schema "schema = Schema([ColumnSchema(:x, :continuous, Float64, false, :feature)])"
@_example ColumnSchema "column = ColumnSchema(:age, :continuous, Float64, false, :feature)"
@_example Impute "transformer = Impute(strategy=:mean)"
@_example OneHotEncode "transformer = OneHotEncode(handle_unknown=:ignore)"
@_example ColumnTable "table = column_table((x=[1, 2], y=[:a, :b]))"
@_example CategoricalColumn "column = Tilia.categorical_column([:a, :b, :a])"
@_example column_table "table = column_table((x=[1, 2],))"

@_example LinearRegression "model = LinearRegression(solver=:qr)"
@_example RidgeRegression "model = RidgeRegression(lambda=1.0)"
@_example LogisticRegression "model = LogisticRegression(lambda=1.0)"
@_example Lasso "model = Lasso(lambda=0.1)"
@_example ElasticNet "model = ElasticNet(lambda=0.1, l1_ratio=0.5)"
@_example SparseLogisticRegression "model = SparseLogisticRegression(lambda=0.1)"
@_example PCA "model = PCA(n_components=2)"
@_example TruncatedSVD "model = TruncatedSVD(n_components=2)"
@_example KMeans "model = KMeans(n_clusters=3)"
@_example GaussianNaiveBayes "model = GaussianNaiveBayes()"
@_example LinearDiscriminantAnalysis "model = LinearDiscriminantAnalysis()"
@_example QuadraticDiscriminantAnalysis "model = QuadraticDiscriminantAnalysis()"
@_example NearestNeighbors "index = fit(NearestNeighbors(n_neighbors=3), randn(10, 2))"
@_example KNeighborsClassifier "model = KNeighborsClassifier(n_neighbors=3)"
@_example KNeighborsRegressor "model = KNeighborsRegressor(n_neighbors=3)"
@_example kneighbors "distances, indices = kneighbors(fit(NearestNeighbors(), randn(5, 2)), randn(2, 2))"
@_example GaussianMixture "model = GaussianMixture(n_components=2)"
@_example DecisionTreeClassifier "model = DecisionTreeClassifier(max_depth=3)"
@_example DecisionTreeRegressor "model = DecisionTreeRegressor(max_depth=3)"
@_example RandomForestClassifier "model = RandomForestClassifier(n_estimators=20)"
@_example RandomForestRegressor "model = RandomForestRegressor(n_estimators=20)"
@_example ExtraTreesClassifier "model = ExtraTreesClassifier(n_estimators=20)"
@_example ExtraTreesRegressor "model = ExtraTreesRegressor(n_estimators=20)"
@_example HistGradientBoostingClassifier "model = HistGradientBoostingClassifier(n_estimators=20)"
@_example HistGradientBoostingRegressor "model = HistGradientBoostingRegressor(n_estimators=20)"
@_example IsolationForest "model = IsolationForest(n_estimators=20)"
@_example anomaly_score "scores = anomaly_score(fit(IsolationForest(n_estimators=2), randn(8, 2)), randn(2, 2))"
@_example KernelRidgeRegression "model = KernelRidgeRegression(kernel=:rbf)"
@_example SupportVectorClassifier "model = SupportVectorClassifier(kernel=:linear)"
@_example SupportVectorRegressor "model = SupportVectorRegressor(kernel=:linear)"
@_example MLPClassifier "model = MLPClassifier(hidden_units=16)"
@_example MLPRegressor "model = MLPRegressor(hidden_units=16)"
@_example BernoulliRBM "model = BernoulliRBM(n_components=8)"

@_example capabilities "capabilities(LogisticRegression())"
@_example input_contract "input_contract(Standardize())"
@_example output_schema "output_schema(Standardize(), Schema([ColumnSchema(:x, :continuous, Float64, false, :feature)]))"
@_example model_catalog "model_catalog(task=:classification, probabilistic=true)"
@_example accuracy_score "accuracy_score([:a, :b], [:a, :a])"
@_example precision_score "precision_score([:a, :b], [:a, :a])"
@_example recall_score "recall_score([:a, :b], [:a, :a])"
@_example f1_score "f1_score([:a, :b], [:a, :a])"
@_example confusion_matrix "confusion_matrix([:a, :b], [:a, :a])"
@_example log_loss "log_loss([1, 2], [0.8 0.2; 0.1 0.9])"
@_example mean_squared_error "mean_squared_error([1.0, 2.0], [1.0, 3.0])"
@_example root_mean_squared_error "root_mean_squared_error([1.0, 2.0], [1.0, 3.0])"
@_example train_test_split "train_test_split(randn(10, 2), randn(10); test_size=0.2)"
@_example KFold "cv = KFold(5; shuffle=true, seed=42)"
@_example split "folds = split(KFold(3), 12)"
@_example cross_validate "cross_validate(RidgeRegression(), randn(12, 2), randn(12); cv=KFold(3))"
