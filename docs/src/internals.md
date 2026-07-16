# Internals

This page describes implementation structure for contributors and advanced
inspection. APIs explicitly listed as internal may change without the stability
expected from exported user operations.

## Layered architecture

Tilia separates five concerns:

1. **Data and schema:** matrix/table adaptation, categorical storage, targets,
   weights, and ordered semantic metadata.
2. **Estimator semantics:** immutable specifications, fitted state,
   capabilities, contracts, and reports.
3. **Semantic graphs:** user-meaningful transforms, branches, leakage
   boundaries, prediction nodes, and schemas.
4. **Numerical execution:** kernels, solvers, primitive lowering, buffers,
   placement, transfers, and backend compilation.
5. **Workflow services:** metrics, selection, inspection, persistence, and
   optional visualization or differentiation.

Statistical model definitions do not own duplicate versions of shared
numerical primitives, and optional backends do not redefine estimator
semantics.

## Semantic and numerical graphs

`Chain`, `Parallel`, and `ColumnMap` lower to explicit semantic nodes and edges.
Roots consume external input; joins consume ordered predecessor tuples.
Contracts record whether a node learns state, consumes the target, changes row
or feature count, supports sparse or missing data, and is valid at inference.

Fit and inference can also be represented as a `NumericalExecutionGraph`.
Semantic nodes expand into primitives such as reductions, centering, matrix
multiplication, stable probability normalization, factorization, or solver
loops. Numerical nodes record shape, element type, representation, device,
buffer assignment, lifetime, aliasing, and mutability.

This separation allows a pipeline to remain meaningful to users while a
backend operates on lower-level numerical regions.

## CPU execution and optimization

The CPU interpreter is authoritative. Ordinary fitted-graph inference executes
the fitted node plan directly without rebuilding detailed lowering metadata on
every call. Linear chains retain a dedicated hot path; dependency-driven graph
execution handles branches and joins.

Semantic optimization includes constant folding, redundant conversion
elimination, dead-node elimination, and compatible transform fusion. Execution
planning computes topological order, last use, reusable logical buffers, and
peak buffer count. Device placement assigns nodes and makes cross-device
transfers explicit; adjacent duplicate transfer declarations are coalesced.

## Inspection helpers

Advanced internal helpers include:

```julia
Tilia.graph_data(fitted)
Tilia.trace(fitted, X; operation=:predict)
Tilia.execution_plan(fitted.graph)
Tilia.lower_graph(fitted.graph, X; phase=:inference)
Tilia.device_placement(fitted.graph; default=:cpu)
```

`graph_data` returns backend-neutral node and edge metadata. `trace` executes a
fitted graph while recording per-node operation, input/output shapes, elapsed
nanoseconds, and output size. Lowering and placement APIs are primarily for
backend and compiler work.

## Randomness and reports

`FitContext` carries a root seed and stream identifier. Composite workflows
derive deterministic named substreams for folds, trials, estimators, trees,
restarts, or inspection repetitions. Reports retain this identity alongside
thread count, backend, warnings, numerical policy, and operation-specific
details.

## Persistence boundary

Structural persistence encodes supported Tilia types, tuples, named tuples,
schemas, scalars, and typed numeric arrays. The decoder accepts known
structural names and versions rather than evaluating arbitrary Julia types.
Format migrations are explicit functions between version representations.

## Advanced namespaces

`Tilia.Kernels` contains reusable reductions, statistics, distances,
probability operations, losses, and metrics. `Tilia.Solvers` contains reusable
optimization and factorization protocols. Contributors should prefer these
namespaces to local numerical reimplementations.

## Core public API

```@autodocs
Modules = [Tilia]
Private = false
```

## Numerical kernels

```@autodocs
Modules = [Tilia.Kernels]
Private = false
```

## Solvers

```@autodocs
Modules = [Tilia.Solvers]
Private = false
```
