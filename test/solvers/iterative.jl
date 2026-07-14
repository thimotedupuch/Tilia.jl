@testset "Iterative optimization solvers" begin
    A = [4.0 1; 1 3]
    b = [1.0, 2]
    cg = Tilia.Solvers.conjugate_gradient(A, b; tolerance=1e-12)
    @test cg.converged
    @test cg.parameters ≈ A \ b atol=1e-10
    @test all(diff(cg.objective_history) .<= 1e-12)

    rectangular = [1.0 0; 0 1; 1 1; 2 1]
    target = rectangular * [2.0, -1.0]
    lsqr_result = Tilia.Solvers.lsqr(rectangular, target; tolerance=1e-10,
                                    max_iterations=20)
    @test lsqr_result.parameters ≈ [2.0, -1.0] atol=1e-8
    @test lsqr_result.residual_norm < 1e-8

    objective = parameters -> 0.5 * dot(parameters, A * parameters) - dot(b, parameters)
    function gradient!(destination, parameters)
        destination .= A * parameters .- b
    end
    lbfgs = Tilia.Solvers.lbfgs(objective, gradient!, zeros(2); tolerance=1e-10)
    @test lbfgs.parameters ≈ A \ b atol=1e-7
    @test all(diff(lbfgs.objective_history) .<= 1e-10)

    newton = Tilia.Solvers.newton_cg(objective, gradient!,
        (_, vector) -> A * vector, zeros(2); tolerance=1e-10)
    @test newton.parameters ≈ A \ b atol=1e-8

    identity_objective = parameters -> sum(abs2, parameters) / 2
    identity_gradient!(destination, parameters) = (destination .= parameters)
    function soft_prox!(destination, values, step)
        destination .= sign.(values) .* max.(abs.(values) .- 0.1step, 0)
    end
    proximal = Tilia.Solvers.proximal_gradient(identity_objective,
        identity_gradient!, soft_prox!, [2.0, -1.0]; step_size=0.5, tolerance=1e-8)
    accelerated = Tilia.Solvers.fista(identity_objective,
        identity_gradient!, soft_prox!, [2.0, -1.0]; step_size=0.5, tolerance=1e-8)
    @test norm(proximal.parameters) < 1e-6
    @test norm(accelerated.parameters) < 1e-6

    samples = [1.0, 2.0, 3.0]
    sample_gradient!(destination, parameters, index) =
        (destination[1] = parameters[1] - samples[index])
    sgd = Tilia.Solvers.stochastic_gradient(sample_gradient!, [0.0], 3;
        learning_rate=0.1, epochs=100, objective=p -> abs2(p[1] - 2))
    @test sgd.parameters[1] ≈ 2 atol=0.15

    em = Tilia.Solvers.expectation_maximization(
        parameter -> parameter,
        parameter -> (parameter + 2) / 2,
        parameter -> -(parameter - 2)^2,
        0.0; tolerance=1e-10)
    @test em.parameters ≈ 2 atol=1e-5
    @test all(diff(em.objective_history) .>= -1e-12)

    design = [1.0 0; 1 1; 1 2; 1 3]
    binary = [0.0, 0, 1, 1]
    newton_fit = Tilia.Solvers.binary_logistic_newton(design, binary; lambda=0.1)
    irls_fit = Tilia.Solvers.binary_logistic_irls(design, binary; lambda=0.1)
    @test irls_fit.parameters == newton_fit.parameters
end
