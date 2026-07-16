# Differentiation

Tilia separates differentiable numerical objectives from estimator fit control
flow. Analytic derivatives remain preferred for built-in objectives; automatic
differentiation is an optional bridge for custom scalar objectives rather than
a hard dependency of every model.

## Objective protocol

An objective implements scalar value and in-place gradient operations:

```julia
Tilia.value(objective, parameters, data)
Tilia.gradient!(destination, objective, parameters, data)
Tilia.value_and_gradient!(destination, objective, parameters, data)
Tilia.jvp(objective, parameters, tangent, data)
Tilia.vjp!(destination, objective, parameters, cotangent, data)
```

JVP and VJP use the objective gradient protocol and validate parameter/tangent
dimensions.

## Analytic logistic objective

`BinaryLogisticObjective` evaluates weighted binary logistic loss with a masked
L2 penalty. `LogisticBatch` keeps design, encoded target, and observation
weights together:

```julia
design = [ones(size(X,1)) X]
batch = Tilia.LogisticBatch(design, Float64.(y), ones(size(X,1)))
objective = Tilia.BinaryLogisticObjective(0.1, [0.0; ones(size(X,2))])
parameters = zeros(size(design,2))
gradient = similar(parameters)

loss, gradient = Tilia.value_and_gradient!(
    gradient, objective, parameters, batch,
)
```

The zero penalty mask for the first parameter leaves the intercept
unregularized. Stable logistic expressions are implemented analytically.

## Optional automatic differentiation

Loading [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/)
activates `autodiff_objective`:

```julia
using Tilia
using DifferentiationInterface
using ForwardDiff

objective = Tilia.autodiff_objective(
    (parameters, data) -> sum(abs2, parameters .- data),
    AutoForwardDiff(),
)

parameters = ones(3)
data = fill(0.5, 3)
gradient = similar(parameters)
Tilia.value_and_gradient!(gradient, objective, parameters, data)
```

Without DifferentiationInterface loaded, constructing an AD-backed objective
raises an explanatory error. The chosen AD backend is supplied explicitly by
the application.

## Scope and design guidance

- Use analytic gradients for built-in objectives when feasible.
- Keep mutation, stopping logic, data loading, and reporting outside the
  differentiable scalar function.
- Treat objective data as an explicit argument instead of closing over mutable
  training state.
- Validate AD gradients against finite differences or an analytic reference.
- Do not infer that loading an AD backend makes every estimator differentiable.

The isolated development environment is `test/differentiation`:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/differentiation \
    test/differentiation/runtests.jl
```
