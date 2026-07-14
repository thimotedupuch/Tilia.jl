using Test
using Tilia
using Reactant

@testset "Reactant standardization and logistic inference" begin
    X = Float32[-2 0; -1 1; 1 -1; 2 0]
    y = [:negative, :negative, :positive, :positive]
    model = Chain(Standardize(), LogisticRegression(lambda=1.0f0))
    cpu = fit(model, X, y)
    cache = CompilationCache()
    context = FitContext(backend=ReactantBackend(device=:cpu), cache=cache)
    accelerated = fit(model, X, y; context=context)

    probabilities = predict_proba(accelerated, X)
    @test probabilities ≈ predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    @test predict(accelerated, X) == predict(cpu, X)
    first_report = report(accelerated)
    @test first_report.backend == :reactant
    @test first_report.details.device == :cpu
    @test first_report.details.accelerator_nodes == [1, 2]
    @test first_report.details.host_nodes == Int[]
    @test first_report.details.host_fit_nodes == [1, 2]
    @test first_report.details.compilation_nanoseconds > 0
    @test first_report.details.execution_nanoseconds > 0
    @test first_report.details.transferred_bytes > 0
    @test first_report.details.accelerated_logistic_objective
    @test isfinite(first_report.details.accelerated_objective_value)
    @test isempty(first_report.details.unsupported_operations)
    @test !isempty(first_report.details.fallback_operations)

    predict_proba(accelerated, X)
    @test report(accelerated).details.compilation_cache_hits >= 1

    Xnew = Float32[-3 0; 0 0; 3 0]
    @test predict_proba(accelerated, Xnew) ≈ predict_proba(cpu, Xnew) rtol=1.0f-4 atol=1.0f-5
end

@testset "Reactant fallback semantics" begin
    X = Float32[1 2; 3 4; 5 6]
    y = Float32[1, 2, 3]
    unsupported = Chain(Standardize(), MeanRegressor())
    @test_throws Tilia.UnsupportedBackendError fit(unsupported, X, y;
        context=FitContext(backend=ReactantBackend()))

    fallback = fit(unsupported, X, y;
        context=FitContext(backend=ReactantBackend(fallback=:cpu)))
    fallback_report = report(fallback)
    @test fallback_report.backend == :cpu
    @test isempty(fallback_report.details.accelerator_nodes)
    @test fallback_report.details.host_nodes == [1, 2]
    @test !isempty(fallback_report.details.unsupported_operations)
    @test !isempty(fallback_report.details.fallback_operations)
    @test predict(fallback, X) == fill(2.0f0, 3)
end
