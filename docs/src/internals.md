# Internals

The semantic graph records user-meaningful operations and leakage boundaries.
Numerical kernels and solvers are separate from statistical model definitions.
The CPU interpreter is authoritative; optional backends lower only supported
regions and report transfers or fallbacks.

Fitting records semantic and numerical execution graphs. Ordinary CPU
inference executes the fitted node plan directly without rebuilding lowering
metadata on every call; explicit `trace` and internal lowering APIs construct
detailed per-call inspection data when requested.

Graph execution is dependency-driven. `Parallel` is lowered to sibling nodes
and `ColumnMap` to explicit selection, transformation, and merge nodes. Roots
consume the external input, joins consume ordered tuples of predecessor
outputs, and reports therefore expose branch-level schemas, timings, buffers,
and numerical primitives. Linear chains retain a dedicated inference hot path.

Useful advanced namespaces are `Tilia.Kernels` and `Tilia.Solvers`. Internal
graph inspection is available through `Tilia.graph_data`, `Tilia.trace`,
`Tilia.execution_plan`, and `Tilia.device_placement`.

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
