# Data and schemas

Tilia accepts dense and sparse matrices, Tables.jl-compatible sources, its
owned `ColumnTable`, and `Dataset`. Regardless of representation, observations
are rows and feature order is significant.

## Numeric matrices

Use a matrix when all features are already numeric and share a suitable element
type:

```julia
X = Float32[
    20  35_000
    40  72_000
    55  91_000
]

fitted = fit(Standardize(), X)
transform(fitted, X)
```

Most numeric estimators require finite values. Missing values, `NaN`, and
infinities must be handled explicitly rather than being silently propagated.
Sparse support varies by estimator; inspect `capabilities(model).sparse`.

## Tables and owned column storage

Any source implementing [Tables.jl](https://tables.juliadata.org/stable/) can
be converted to Tilia's column-oriented representation:

```julia
source = (
    age=[20.0, 40.0, 55.0],
    color=[:blue, :red, :blue],
    income=[35_000.0, 72_000.0, 91_000.0],
)

table = column_table(source)
table.names
table.schema
```

`column_table` copies columns into Tilia-owned storage. Numeric columns remain
ordinary vectors. Categorical columns receive deterministic pools and integer
codes, with their level order recorded in the schema.

`ColumnTable` itself implements Tables.jl column access, so it can be passed to
other compatible tooling without converting it back to a matrix.

## What a schema records

A `Schema` is ordered metadata, not only a feature count. Every `ColumnSchema`
records:

| Field | Meaning |
|:--|:--|
| `name` | Stable semantic column name |
| `logical_type` | For example, `:continuous` or `:categorical` |
| `physical_type` | Stored Julia element type, excluding `Missing` |
| `allows_missing` | Whether missing input was observed or declared |
| `role` | Feature, prediction, or another semantic role |
| `levels`, `ordered` | Categorical pool and ordering |
| `unknown_policy`, `missing_policy` | Transformation behavior for exceptional values |
| `provenance`, `generated_name` | Source columns and names for generated features |

At the schema level, target name, target logical and physical type, missingness,
and fitted classification order are also retained. Fitted estimators validate
new inputs against the relevant parts of this contract.

```julia
for column in table.schema.columns
    println(column.name, " → ", column.logical_type)
end
```

## Dataset: features, target, and weights

`Dataset` keeps the supervised pieces together:

```julia
labels = [:low, :medium, :high]
weights = [1.0, 0.5, 2.0]

dataset = Dataset(table; target=labels, weights=weights)
fitted = fit(
    Chain(
        ColumnMap(
            :age => Standardize(),
            :income => RobustScale(),
            :color => OneHotEncode(passthrough_numeric=false),
        ),
        LogisticRegression(),
    ),
    dataset,
)
```

The constructor validates row counts and enriches the schema with target
metadata. Weight support is estimator-specific and declared by
`capabilities(model).weights`; unsupported weights are rejected.

## Missing and categorical values

Missingness and categorical conversion are explicit pipeline operations:

```julia
preprocess = ColumnMap(
    (:age, :income) => Chain(Impute(), Standardize()),
    :color => Chain(Impute(), OneHotEncode(passthrough_numeric=false)),
)
```

Fit imputation before encoding when categorical values may be missing.
`OneHotEncode` learns level order during fitting, generates deterministic
output names, and records source-column provenance. Unknown-category policy is
part of the categorical metadata rather than an undocumented prediction-time
choice.

## Schema propagation through graphs

Pipeline construction can derive semantic output schemas for supported steps:

```julia
input = table.schema
encoded = output_schema(
    ColumnMap(
        (:age, :income) => Standardize(),
        :color => OneHotEncode(passthrough_numeric=false),
    ),
    input,
)
```

Generated polynomial terms, decomposition components, encoded levels,
selected columns, and concatenated branches retain names and provenance. This
allows graph validation to detect incompatible compositions before treating
the whole pipeline as an opaque matrix function.

## Practical rules

1. Keep observations in rows at every boundary.
2. Prefer a Tables.jl source when names and categorical meaning matter.
3. Convert missingness and categories explicitly in a fitted pipeline.
4. Preserve the fitted feature order at prediction time.
5. Inspect schemas and capabilities instead of relying on element types alone.
