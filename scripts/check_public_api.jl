#!/usr/bin/env julia

using Tilia

"""Return missing public docstrings, examples, and nonempty example bodies."""
function check_public_api()
    undocumented = Docs.undocumented_names(Tilia; private=false)
    public_bindings = Set{Symbol}()
    for name in names(Tilia; all=false, imported=false)
        name in (:Tilia, :Kernels, :Solvers) && continue
        value = getfield(Tilia, name)
        (value isa Function || value isa Type || value isa UnionAll) &&
            push!(public_bindings, name)
    end
    missing_examples = sort!(collect(setdiff(
        public_bindings, Tilia.PUBLIC_EXAMPLE_NAMES)))
    empty_examples = sort!([name for name in public_bindings
                            if haskey(Tilia.PUBLIC_EXAMPLES, name) &&
                               isempty(Tilia.PUBLIC_EXAMPLES[name])])
    (; undocumented, missing_examples, empty_examples)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = check_public_api()
    all(isempty, values(result)) || error("public API check failed: $result")
    println("Public API documentation and examples are complete.")
end
