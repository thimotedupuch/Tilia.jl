# TiliaMakieRecipes

Makie plotting recipes for Tilia's semantic result types. This is a separate
package so that Tilia's core package has no plotting dependency or extension.

Load it alongside Tilia and Makie to enable plotting:

```julia
using Tilia, TiliaMakieRecipes, CairoMakie

result = confusion_matrix([:no, :yes], [:no, :yes])
plot(result)
```

For a local checkout, instantiate and run the isolated tests with:

```sh
julia --project=TiliaMakieRecipes/test -e 'using Pkg; Pkg.instantiate()'
julia --project=TiliaMakieRecipes/test TiliaMakieRecipes/test/runtests.jl
```
