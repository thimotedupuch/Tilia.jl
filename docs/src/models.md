# Models

Foundational models include linear, ridge, logistic, Lasso, and elastic-net
estimators, plus incremental mini-batch SGD classification and regression.
Adaptive hinge-spline regression is available through `MARSRegressor`.
Supervised latent reduction is available through
`PartialLeastSquaresRegression`; call `transform` on its fitted state to obtain
the target-correlated score representation.
Native preprocessing includes standard, min–max, robust, row normalization,
and polynomial feature transforms. Decomposition and probabilistic families include PCA, truncated
SVD, nonnegative matrix factorization, random projection, FastICA, k-means, DBSCAN,
agglomerative clustering,
feature agglomeration,
Gaussian mixtures, Gaussian and multinomial naive Bayes, LDA, and QDA. Exact neighbors,
CART, forests, extra trees, histogram boosting, and isolation forest cover
pairwise and branch-heavy workloads. Kernel ridge, support-vector estimators,
shallow MLPs, and Bernoulli RBMs complete the initial model set.

Histogram boosting uses regularized second-order leaf updates and supports
deterministic row and feature subsampling without exposing an XGBoost
compatibility facade.

```julia
clusters = fit(KMeans(n_clusters=3), X)
labels = predict(clusters, X)

classifier = fit(RandomForestClassifier(n_estimators=100), X, y)
probabilities = predict_proba(classifier, Xnew)
```

Inspect `capabilities(model)` before relying on sparse, weighted,
probabilistic, or partial-fit behavior.
