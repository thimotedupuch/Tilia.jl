# Differentiation

Analytic derivatives remain preferred. `BinaryLogisticObjective` supplies
value, gradient, JVP, and VJP operations directly. Loading
DifferentiationInterface enables custom scalar objectives without making AD a
hard dependency.

```julia
using Tilia, DifferentiationInterface, ForwardDiff
objective = Tilia.autodiff_objective(x -> sum(abs2, x), AutoForwardDiff())
gradient = zeros(3)
Tilia.gradient!(gradient, objective, ones(3))
```
