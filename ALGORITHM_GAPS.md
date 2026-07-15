# Temporary algorithm implementation checklist

This is a deliberately bounded list of broadly useful gaps, not a promise to
clone every scikit-learn estimator. Delete this file only when every checkbox is
implemented, documented, persisted where applicable, and covered by conformance
and numerical-reference tests.

- [x] `RobustScale`: median/IQR scaling for outlier-heavy numeric data.
- [ ] `SGDClassifier` and `SGDRegressor`: bounded-memory linear learning and
      `partial_fit` for large or streaming datasets.
- [x] `MultinomialNaiveBayes`: fast sparse/count-data classification.
- [ ] `DBSCAN`: density clustering with explicit noise labels and no required
      cluster count.
- [ ] `AgglomerativeClustering`: deterministic hierarchical clustering for
      small and medium datasets.
- [ ] `NMF`: interpretable nonnegative low-rank decomposition.
- [ ] `MARSRegressor`: adaptive hinge-spline regression with pruning for smooth
      nonlinear tabular relationships.
- [ ] Strengthen native histogram boosting with XGBoost-relevant capabilities:
      second-order leaf updates, L1/L2 regularization, and deterministic row and
      feature subsampling. Do not add an `XGBoost` compatibility facade.

Existing forests, SVMs, nearest neighbors, PCA, Gaussian mixtures, MLPs, and
graph composition already cover several adjacent use cases. Their sklearn
variants are therefore not duplicated here merely for API breadth.
