# Documentation sources

The Markdown files under `docs/src` remain readable directly. Build the full
manual from the repository root with the separate documentation environment:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
