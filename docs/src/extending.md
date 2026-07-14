# Extending Tilia

New estimators should define an immutable specification, fitted state,
capabilities, validation, fitting, inference, reporting, persistence, reference
fixtures, conformance tests, and benchmarks. Reuse `Tilia.Kernels` and
`Tilia.Solvers`; do not embed duplicate numerical primitives in models.

```julia
struct MyRegressor <: Tilia.AbstractPredictor end
Tilia.capabilities(::Type{MyRegressor}) =
    (task=:regression, sparse=false, missing=false, weights=false,
     partial_fit=false, probabilistic=false)
```
