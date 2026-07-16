ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1

using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.add(name = "CairoMakie", version = "0.15"; preserve = Pkg.PRESERVE_ALL)
Pkg.add(name = "MLDatasets"; preserve = Pkg.PRESERVE_ALL)
Pkg.add(name = "DataFrames"; preserve = Pkg.PRESERVE_ALL)
Pkg.instantiate()
Pkg.precompile()
