# Persistence

Tilia persistence stores supported fitted models structurally. It does not use
Julia `Serialization`, embed executable code, or depend on object-memory
layout.

```julia
fitted = fit(model, X, y)
save_model("model-artifact", fitted)

restored = load_model("model-artifact")
predict(restored, Xnew)
```

## Artifact layout

`save_model` writes a directory containing:

- a versioned TOML structural manifest;
- typed binary array payloads;
- dimensions, element types, and endianness metadata;
- checksums for corruption detection;
- package and Julia-version metadata;
- estimator schema version and migration history.

Schemas, class order, graph structure, fitted node state, reports, and supported
numeric arrays are encoded as data. The artifact is intended to be inspectable
and migratable rather than an opaque Julia object dump.

## Verification during loading

`load_model` rejects:

- unknown or future format versions;
- unknown structural types or fields;
- unsupported scalar or array element types;
- array dimension and payload inconsistencies;
- host/artifact endianness mismatches;
- checksum corruption and unexpected trailing data.

These failures use persistence-specific errors instead of returning a partly
decoded model.

## Format versions and migration

The current format includes explicit version migration. Version-one manifests
can be represented as version two in memory while retaining a migration history;
payload arrays do not need to be rewritten for that migration.

Format migration is different from arbitrary model-code compatibility. A new
or changed estimator still needs an intentional structural persistence contract
and tests. Do not assume an external custom estimator becomes persistable only
because it implements `fit` and `predict`.

## Graphs and optional backends

Fitted semantic graphs persist their graph, fitted nodes, schemas, and report
state. A fitted Reactant graph saves its authoritative CPU graph; compiled
executables, device buffers, and compilation cache entries are deliberately not
portable artifact content.

## Operational recommendations

1. Save into a new path and retain the previous known-good artifact until the
   new one has been loaded and checked.
2. Record the application version and training-data identity outside the model
   when those are deployment requirements.
3. Load untrusted artifacts only under the same trust policy used for other
   application data, even though the format does not deserialize executable
   Julia objects.
4. Test predictions before and after a round trip for every deployed model
   family.
5. Keep artifacts with reference inputs and expected outputs when long-term
   compatibility matters.

```julia
before = predict(fitted, reference_X)
save_model("candidate", fitted)
after = predict(load_model("candidate"), reference_X)
@assert before == after
```
