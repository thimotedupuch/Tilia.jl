using Test
using Tilia
using DifferentiationInterface
using ForwardDiff

@testset "DifferentiationInterface custom objectives" begin
    function_value = (parameters, data) ->
        sum(abs2, parameters .- data.center) + data.scale * sum(parameters)
    backend = AutoForwardDiff()
    objective = Tilia.autodiff_objective(function_value, backend)
    parameters = [1.0, -2.0, 3.0]
    data = (center=[0.5, -1.0, 2.0], scale=0.25)
    expected = 2 .* (parameters .- data.center) .+ data.scale

    destination = zeros(3)
    @test Tilia.gradient!(destination, objective, parameters, data) === destination
    @test destination ≈ expected atol=1e-12
    objective_value, returned = Tilia.value_and_gradient!(
        destination, objective, parameters, data)
    @test returned === destination
    @test objective_value ≈ function_value(parameters, data)
    @test destination ≈ expected atol=1e-12
    @test Tilia.jvp(objective, parameters, ones(3), data) ≈ sum(expected)

    cotangent = zeros(3)
    Tilia.vjp!(cotangent, objective, parameters, 2.0, data)
    @test cotangent ≈ 2 .* expected
end
