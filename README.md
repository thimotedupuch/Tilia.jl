# Tilia.jl

Tilia is an experimental Julia-native classical machine-learning stack. The
current checkpoint establishes the core estimator contracts and a semantic
graph interpreter.

```julia
using Tilia

X = [1.0 10.0; 2.0 20.0; 3.0 30.0]
y = [2.0, 4.0, 9.0]

model = Chain(Standardize(), MeanRegressor())
fitted = fit(model, X, y)
predict(fitted, X)
report(fitted)
```

Observations are rows and features are columns. Estimator specifications are
immutable; fitting returns a separate object containing learned state,
training schema, and a structured report.

## Optional Reactant execution

Reactant is a weak dependency and is not installed or loaded by the core
package. In an environment containing Reactant, dense standardization and
logistic inference can be compiled explicitly:

```julia
using Tilia, Reactant

context = FitContext(backend=ReactantBackend(device=:cpu))
accelerated = fit(Chain(Standardize(), LogisticRegression()), X, y;
                  context=context)
predict_proba(accelerated, Xnew)
report(accelerated)
```

The report records compilation and execution time, transferred bytes, device
placement, cache hits, unsupported operations, and fallbacks. The prototype
accelerates standardization, matrix multiplication, probabilities, and the
logistic objective. Fit-time statistics and Newton iterations remain on CPU
and are reported as such. `ReactantBackend(fallback=:cpu)` opts into an
explicit host fallback; the default is to raise `UnsupportedBackendError`.

Accelerator tests use their own environment:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/accelerator \
  -e 'using Pkg; Pkg.instantiate(); include("test/accelerator/runtests.jl")'
```

## Decomposition, clustering, and probabilistic models

The CPU backend includes deterministic PCA and truncated SVD, Lloyd k-means,
Gaussian mixtures, Gaussian naive Bayes, linear and quadratic discriminant
analysis, and exact nearest-neighbor estimators. Learned state remains separate
from immutable specifications and every fit produces a structured report.

```julia
X = [-2.0 -1.0; -1.0 -2.0; 1.0 2.0; 2.0 1.0]
y = [:negative, :negative, :positive, :positive]

pipeline = Chain(PCA(n_components=1),
                 KNeighborsClassifier(n_neighbors=1))
fitted = fit(pipeline, X, y)
predict(fitted, X)
predict_proba(fitted, X)

clusters = fit(KMeans(n_clusters=2), X)
clusters.centers
report(clusters).details.objective_history

mixture = fit(GaussianMixture(n_components=2), X)
predict_proba(mixture, X)
```

PCA centers inputs while `TruncatedSVD` deliberately does not. Rows are always
observations. K-means and mixture component numbers are one-based; component
ordering is deterministic for a fixed `FitContext` RNG but has no semantic
class meaning.

Offline reference fixtures for these models were generated with scikit-learn
and are stored under `test/reference`; Python is not used by Tilia or its Julia
test suite.

## Optional differentiation and visualization

`DifferentiationInterface` and Makie are weak dependencies. Loading them adds
automatic differentiation for custom scalar objectives and direct plotting of
confusion matrices, ROC results, cross-validation scores, and optimization
traces. Their isolated test environments live in `test/differentiation` and
`test/visualization`.

## Benchmarks and documentation

The persistent benchmark environment is `benchmark/`. It uses the local Tilia
checkout and keeps kernel and model benchmarks separate:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/kernels/runbenchmarks.jl
julia --project=benchmark benchmark/models/runbenchmarks.jl
```

Workflow-oriented documentation starts at `docs/src/index.md` and covers data,
graphs, models, metrics, selection, acceleration, differentiation,
persistence, numerical behavior, extension, and internals.
