# Pipelines and graphs

`Chain` fits every transformation only on training rows. `Select`, `ColumnMap`,
`Parallel`, and `Concatenate` express column and branch structure explicitly.

```julia
features = ColumnMap(:age => Standardize(),
                     :color => OneHotEncode(passthrough_numeric=false))
model = Chain(features, LogisticRegression())
fitted = fit(model, table, labels)
```

The CPU interpreter validates topology and leakage, records per-node timings,
and supports tracing. Optimization includes constant folding, dead-node and
conversion elimination, affine fusion, device placement, transfer coalescing,
and buffer planning.
