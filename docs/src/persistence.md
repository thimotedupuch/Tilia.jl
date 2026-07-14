# Persistence

`save_model` writes a versioned directory containing TOML schemas, typed binary
arrays, checksums, package metadata, and Julia-version metadata. It never uses
Julia `Serialization`.

```julia
save_model("model-artifact", fitted)
restored = load_model("model-artifact")
predict(restored, Xnew)
```

Loading rejects unknown format versions, unknown structural types, endianness
mismatches, and checksum corruption.
