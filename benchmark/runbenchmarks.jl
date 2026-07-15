for suite in ("compilation", "kernels", "training", "inference", "memory", "graph",
              "cpu_scaling", "sparse", "preprocessing")
    println((suite=suite,))
    include(joinpath(@__DIR__, suite, "runbenchmarks.jl"))
end

println((suite=:accelerator_scaling, command=
    "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=test/accelerator benchmark/accelerator/runbenchmarks.jl"))
