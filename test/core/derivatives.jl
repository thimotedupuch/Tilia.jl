@testset "Analytic derivative protocol" begin
    design = [1.0 0.0 1.0; 1.0 1.0 1.0; 1.0 2.0 1.0; 1.0 3.0 1.0]
    target = [0.0, 0.0, 1.0, 1.0]
    weights = [1.0, 2.0, 1.0, 2.0]
    batch = Tilia.LogisticBatch(design, target, weights)
    objective = Tilia.BinaryLogisticObjective(0.5, [1.0, 1.0, 0.0])
    parameters = [0.2, -0.4, 0.1]
    gradient = zeros(3)

    objective_value, returned = Tilia.value_and_gradient!(
        gradient, objective, parameters, batch)
    @test returned === gradient
    @test objective_value == Tilia.value(objective, parameters, batch)
    step = 1e-6
    finite_difference = [(Tilia.value(objective, parameters .+ step .* (1:3 .== index), batch) -
                          Tilia.value(objective, parameters .- step .* (1:3 .== index), batch)) /
                         (2step) for index in 1:3]
    @test gradient ≈ finite_difference atol=1e-8

    tangent = [2.0, -1.0, 0.5]
    @test Tilia.jvp(objective, parameters, tangent, batch) ≈ dot(gradient, tangent)
    destination = similar(parameters)
    @test Tilia.vjp!(destination, objective, parameters, 3.0, batch) === destination
    @test destination ≈ 3 .* gradient

    @test_throws DimensionMismatch Tilia.LogisticBatch(design, target[1:3], weights)
    @test_throws DimensionMismatch Tilia.gradient!(zeros(2), objective, parameters, batch)
end
