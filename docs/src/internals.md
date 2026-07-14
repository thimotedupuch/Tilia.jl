# Internals

The semantic graph records user-meaningful operations and leakage boundaries.
Numerical kernels and solvers are separate from statistical model definitions.
The CPU interpreter is authoritative; optional backends lower only supported
regions and report transfers or fallbacks.

Useful advanced namespaces are `Tilia.Kernels` and `Tilia.Solvers`. Internal
graph inspection is available through `Tilia.graph_data`, `Tilia.trace`,
`Tilia.execution_plan`, and `Tilia.device_placement`.
