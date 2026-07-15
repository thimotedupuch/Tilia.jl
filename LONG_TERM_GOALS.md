# Long-term credibility goals

Tilia should become a dependable Julia system for inspectable classical machine
learning. Credibility should come from correctness, coherent execution,
performance evidence, and operational reliability—not from copying every API or
reimplementing every specialist algorithm.

## 1. Make correctness independently convincing

- Maintain numerical reference tests against published formulas and mature
  implementations where semantics genuinely match.
- Add adversarial coverage for missing values, sparse inputs, degenerate data,
  extreme magnitudes, class imbalance, weights, and rank-deficient problems.
- Define estimator-specific statistical invariants and property tests.
- Publish supported numerical tolerances and determinism guarantees.
- Run compatibility tests across supported Julia versions, operating systems,
  architectures, BLAS implementations, and thread counts.

Credibility signal: users can determine exactly what is guaranteed, and the
test evidence is reproducible outside the project repository.

## 2. Establish performance with honest benchmarks

- Maintain versioned benchmarks for fitting, inference, memory, compilation,
  sparse data, tables, and graph overhead.
- Compare against relevant baselines such as scikit-learn, MLJ-backed models,
  and specialist libraries only when the algorithms and settings are
  comparable.
- Report dataset dimensions, hardware, thread counts, warm-up policy,
  allocations, accuracy, and solver convergence—not latency alone.
- Track regressions in kernels, estimators, and complete workflows.
- Optimize common small and medium tabular workloads before pursuing impressive
  but unrepresentative large benchmarks.

Credibility signal: performance claims are narrow, reproducible, and include
cases where Tilia is slower.

## 3. Develop the graph into a real execution advantage

- Schedule independent branches concurrently when the workload justifies the
  overhead.
- Cache reusable fitted and transformed subgraphs with explicit invalidation.
- Fuse compatible transforms and kernels while preserving semantic reports.
- Plan buffer reuse using actual shapes, aliases, lifetimes, and memory limits.
- Partition graphs across CPU and accelerator backends with visible transfer
  costs and explicit fallback behavior.
- Support reusable subgraphs, multiple outputs, and safe user-defined graph
  nodes.
- Preserve identical prediction semantics between interpreted and optimized
  execution.

Credibility signal: realistic branched workflows become measurably faster or
more memory-efficient, while remaining inspectable and reproducible.

## 4. Offer excellent tabular and schema semantics

- Preserve column names, logical types, categorical levels, roles, and
  provenance through every compatible transform.
- Expand native handling of missing, categorical, ordinal, sparse, count,
  datetime, and text-derived features.
- Detect schema drift at inference and produce actionable diagnostics.
- Define policies for unseen levels, reordered columns, changed physical types,
  and nullable data.
- Integrate naturally with the Tables.jl ecosystem without unnecessary copies.

Credibility signal: heterogeneous production tables are safer and easier to
reason about than anonymous matrices.

## 5. Complete the everyday model-selection workflow

- Add stratified, grouped, repeated, time-series, and nested resampling.
- Support conditional and typed hyperparameter spaces.
- Add efficient random, successive-halving, and Bayesian search where justified.
- Make preprocessing leakage impossible during evaluation and tuning.
- Provide first-class benchmark experiments across models, datasets, metrics,
  and seeds, with resumable results.
- Add threshold selection, probability calibration estimators, learning curves,
  validation curves, and robust uncertainty summaries.

Credibility signal: a practitioner can move from raw table to defensible model
selection without assembling a second framework.

## 6. Strengthen interpretation and diagnostics

- Provide model-native importance where statistically meaningful.
- Add partial-dependence and accumulated-local-effect analysis.
- Support calibrated residual, influence, convergence, and data-quality
  diagnostics.
- Integrate a trustworthy SHAP implementation or specialist backend rather than
  providing a weak substitute.
- Attach explanations to schema provenance so derived features remain
  understandable.

Credibility signal: reports help users discover invalid assumptions, not merely
produce attractive plots.

## 7. Interoperate with specialist models

- Define stable adapters that let external Julia models participate in Tilia
  schemas, graphs, evaluation, reports, and persistence.
- Prioritize high-quality integration with specialist boosted-tree libraries
  over recreating all of XGBoost, LightGBM, or CatBoost.
- Preserve capability metadata and make unsupported operations explicit.
- Keep extension dependencies optional and prevent them from destabilizing the
  core package.

Credibility signal: users can choose a best-in-class learner without abandoning
Tilia's execution and reproducibility model.

## 8. Make persistence production-grade

- Maintain a documented, versioned, migratable, language-neutral artifact
  format where practical.
- Record package versions, numerical policy, schema, capabilities, training
  context, checksums, and relevant environment metadata.
- Define forward-compatibility and deprecation policies.
- Add corruption, migration, and long-lived fixture tests.
- Provide a minimal stable inference surface suitable for services and batch
  jobs.

Credibility signal: a saved model remains auditable and loadable after the
original development environment is gone.

## 9. Build a sustainable extension contract

- Stabilize estimator, fitted-estimator, capability, schema, reporting, and
  lowering interfaces.
- Document how third parties add models without depending on internal fields.
- Add conformance suites that extension authors can run independently.
- Use multiple dispatch where it expresses genuine semantic variation.
- Adopt deprecation windows before breaking public behavior.

Credibility signal: external packages can extend Tilia without coordinated
changes to Tilia's source tree.

## 10. Earn operational trust

- Publish release notes, compatibility bounds, security guidance, and a clear
  stability policy.
- Keep deterministic execution testable across processes and supported thread
  counts.
- Report fallbacks, approximations, convergence failures, and unsupported data
  instead of silently changing behavior.
- Establish reproducible release artifacts and an independent release process.
- Encourage external issue reports, benchmarks, datasets, and downstream usage
  before declaring APIs stable.

Credibility signal: users know which parts are experimental, stable, or
production-ready.

## Deliberate non-goals

- Do not mirror scikit-learn, MLJ, tidymodels, or mlr3 APIs indiscriminately.
- Do not maximize estimator count at the expense of correctness and maintenance.
- Do not claim distributed, GPU, or AutoML capability before it provides a real
  operational advantage.
- Do not hide execution decisions or silently fall back to a different backend.
- Do not turn the core package into a mandatory dependency bundle for every
  external learner.
- A graphical workflow editor may be useful later, but it is not required for
  core technical credibility.

## Suggested maturity gates

### Credible experimental release

- Stable core estimator protocol and artifact format.
- Public benchmark suite with regression tracking.
- Strong conformance coverage for every advertised estimator.
- Reliable mixed-table workflows, evaluation, tuning, and diagnostics.

### Credible general-use release

- Multiple external model integrations.
- Broader resampling and search strategies.
- Demonstrated graph optimization benefits on realistic workloads.
- Compatibility and migration testing across multiple releases.

### Credible production release

- Defined long-term API and artifact compatibility policy.
- Operational deployment guidance and schema-drift handling.
- Mature observability, failure reporting, and security practices.
- Evidence from independent downstream users and workloads.

These gates are evidence requirements, not promises tied to version numbers.
