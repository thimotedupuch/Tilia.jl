@testset "Public API documentation and examples" begin
    @test isempty(Docs.undocumented_names(Tilia; private=false))
    public_bindings = Set{Symbol}()
    for name in names(Tilia; all=false, imported=false)
        name in (:Tilia, :Kernels, :Solvers) && continue
        value = getfield(Tilia, name)
        (value isa Function || value isa Type || value isa UnionAll) && push!(public_bindings, name)
    end
    @test isempty(setdiff(public_bindings, Tilia.PUBLIC_EXAMPLE_NAMES))
    @test all(name -> !isempty(Tilia.PUBLIC_EXAMPLES[name]), public_bindings)
end
