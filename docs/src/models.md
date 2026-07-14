# Models

Foundational models include linear, ridge, logistic, Lasso, and elastic-net
estimators. Decomposition and probabilistic families include PCA, truncated
SVD, k-means, Gaussian mixtures, naive Bayes, LDA, and QDA. Exact neighbors,
CART, forests, extra trees, histogram boosting, and isolation forest cover
pairwise and branch-heavy workloads. Kernel ridge, support-vector estimators,
shallow MLPs, and Bernoulli RBMs complete the initial model set.

```julia
clusters = fit(KMeans(n_clusters=3), X)
labels = predict(clusters, X)

classifier = fit(RandomForestClassifier(n_estimators=100), X, y)
probabilities = predict_proba(classifier, Xnew)
```

Inspect `capabilities(model)` before relying on sparse, weighted,
probabilistic, or partial-fit behavior.
