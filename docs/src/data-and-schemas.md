# Data and schemas

Matrices and Tables.jl sources are accepted. `column_table` copies a table into
Tilia's column-oriented representation and records logical types, physical
types, missingness, roles, and class order.

```julia
table = column_table((age=[20.0, 40.0], color=[:blue, :red]))
table.schema
Dataset(table; target=[:young, :old])
```

Use `Impute` before encoders when missing values are present. Categorical pools
and one-hot output ordering are deterministic.
