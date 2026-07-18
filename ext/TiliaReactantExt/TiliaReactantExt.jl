module TiliaReactantExt

using Tilia
using Reactant
using LinearAlgebra

include("ops/reductions.jl")
include("ops/linear_algebra.jl")
include("ops/indexing.jl")
include("ops/sparse.jl")
include("ops/control_flow.jl")
include("arrays.jl")
include("lowering.jl")
include("compile.jl")
include("cache.jl")
include("backend.jl")
include("diagnostics.jl")

end
