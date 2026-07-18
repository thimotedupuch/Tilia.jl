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
    device_X = Reactant.to_rarray(X)
    device_probabilities = predict_proba(accelerated, device_X; output=:device)
    @test device_probabilities isa Reactant.AbstractConcreteArray
    @test Array(device_probabilities) ≈ predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    @test predict_proba(accelerated, device_X) ≈ predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    device_report = report(accelerated)
    @test device_report.details.transfer_accounting.last_input_location === :device
    @test device_report.details.transfer_accounting.last_output_location === :host
    predict_proba(accelerated, device_X; output=:device)
    device_output_report = report(accelerated)
    @test device_output_report.details.transfer_accounting.last_output_location === :device
    @test device_output_report.details.transfer_accounting.last_result_estimated_bytes == 0
    @test_throws ArgumentError predict_proba(accelerated, X; output=:elsewhere)
    @test_throws Tilia.UnsupportedDataError predict(accelerated, X; output=:device)
    @test predict(accelerated, X) == predict(cpu, X)
    transferred_before_predict = report(accelerated).details.transferred_bytes
    @test predict(accelerated, X) == predict(cpu, X)
    @test report(accelerated).details.transferred_bytes > transferred_before_predict
    probability_bytes = Base.summarysize(predict_proba(cpu, X))
    @test report(accelerated).details.transfer_accounting.last_result_estimated_bytes < probability_bytes
    @test any(location -> location.location === :class_indices,
              report(accelerated).details.transfer_locations)
    first_report = report(accelerated)
    @test first_report.backend == :reactant
    @test first_report.details.device == :cpu
    @test first_report.details.accelerator_nodes == [1, 2]
    @test first_report.details.host_nodes == Int[]
    @test first_report.details.host_fit_nodes == Int[]
    @test first_report.details.accelerator_fit_nodes == [1, 2]
    @test first_report.details.compilation_nanoseconds > 0
    @test first_report.details.execution_nanoseconds > 0
    @test first_report.details.transferred_bytes > 0
    @test first_report.details.phase_placement.fit ==
          [(node_id=1, device=:reactant), (node_id=2, device=:reactant)]
    @test first_report.details.phase_placement.inference ==
          [(node_id=1, device=:reactant), (node_id=2, device=:reactant)]
    timings = first_report.details.phase_timings
    @test timings.compilation_nanoseconds == first_report.details.compilation_nanoseconds
    @test timings.host_conversion_nanoseconds > 0
    @test timings.fit_objective_device_execution_nanoseconds > 0
    @test timings.fit_statistics_device_execution_nanoseconds > 0
    @test timings.fit_statistics_synchronization_and_materialization_nanoseconds > 0
    @test timings.fit_optimizer_device_execution_nanoseconds > 0
    @test timings.fit_optimizer_synchronization_and_materialization_nanoseconds > 0
    @test timings.inference_device_execution_nanoseconds ==
          first_report.details.execution_nanoseconds
    @test timings.inference_synchronization_and_materialization_nanoseconds > 0
    accounting = first_report.details.transfer_accounting
    @test accounting.kind === :estimated_host_summarysize
    @test accounting.estimated_bytes == first_report.details.transferred_bytes
    @test ismissing(accounting.actual_device_transfer_bytes)
    @test first_report.details.accelerated_logistic_objective
    @test isfinite(first_report.details.accelerated_objective_value)
    @test isempty(first_report.details.unsupported_operations)
    @test isempty(first_report.details.fallback_operations)
    statistics = accelerated.cpu_graph.fitted_nodes[1]
    @test report(statistics).backend === :reactant
    @test report(statistics).details.sufficient_statistics_backend === :reactant
    @test report(statistics).details.accumulation_type === Float64
    optimizer = accelerated.cpu_graph.fitted_nodes[2]
    @test report(optimizer).backend === :reactant
    @test report(optimizer).details.optimizer_backend === :reactant
    @test report(optimizer).details.objective_history_kind === :complete
    cpu_history = report(cpu.fitted_nodes[2]).details.objective_history[1]
    accelerator_history = report(optimizer).details.objective_history[1]
    @test length(accelerator_history) == report(optimizer).details.iterations[1]
    @test first(accelerator_history) ≈ first(cpu_history) rtol=1.0f-4 atol=1.0f-5
    @test last(accelerator_history) ≈ last(cpu_history) rtol=1.0f-4 atol=1.0f-5
    @test report(optimizer).details.convergence ==
          report(cpu.fitted_nodes[2]).details.convergence

    predict_proba(accelerated, X)
    @test report(accelerated).details.compilation_cache_hits >= 1

    Xnew = Float32[-3 0; 0 0; 3 0]
    @test predict_proba(accelerated, Xnew) ≈ predict_proba(cpu, Xnew) rtol=1.0f-4 atol=1.0f-5
end

@testset "Reactant bounded shape specialization cache" begin
    X = Float32[-2 0; -1 1; 1 -1; 2 0]
    y = [:negative, :negative, :positive, :positive]
    cache = CompilationCache(max_entries=3)
    accelerated = fit(Chain(Standardize(), LogisticRegression(lambda=1.0f0)),
        X, y; context=FitContext(backend=ReactantBackend(device=:cpu), cache=cache))

    for rows in 1:8
        predict_proba(accelerated, repeat(X[1:1, :], rows, 1))
    end
    cache_report = report(accelerated).details
    @test cache_report.compilation_cache_size <= 3
    @test cache_report.compilation_cache_capacity == 3
    @test cache_report.compilation_cache_evictions > 0
    @test cache_report.compilation_count > cache_report.compilation_cache_size
    memory = cache_report.memory_accounting
    @test memory.portable_model_host_bytes > 0
    @test memory.compilation_cache_host_bytes > 0
    @test memory.retained_device_parameter_bytes == 0
    @test ismissing(memory.retained_executable_device_bytes)
    compilations_before_cleanup = cache_report.compilation_count
    @test empty!(cache) === cache
    @test report(accelerated).details.compilation_cache_size == 0
    @test isempty(cache.order)
    predict_proba(accelerated, X)
    @test report(accelerated).details.compilation_count > compilations_before_cleanup
    @test report(accelerated).details.compilation_cache_size > 0
    @test_throws Tilia.InvalidHyperparameterError CompilationCache(max_entries=0)
end

@testset "Reactant fit sufficient statistics numerics" begin
    X = Float32[1.0f6 1; 1.0f6 + 1 2; 1.0f6 + 2 4;
                1.0f6 + 4 8; 1.0f6 + 8 16; 1.0f6 + 16 32]
    y = Float64[-3, -2, -1, 1, 2, 3]
    weights = Float64[0.25, 1, 2, 4, 8, 16]
    model = Chain(Standardize(), RidgeRegression(lambda=0.5))
    numerics = NumericsPolicy(Float32; accumulation_type=Float64,
                              deterministic_reductions=true)
    cpu = fit(model, X, y; weights=weights,
              context=FitContext(numerics=numerics))
    accelerated = fit(model, X, y; weights=weights, context=FitContext(
        backend=ReactantBackend(device=:cpu), numerics=numerics))

    @test predict(accelerated, X) ≈ predict(cpu, X) rtol=2.0f-4 atol=2.0f-4
    accelerated_statistics = accelerated.cpu_graph.fitted_nodes[1]
    cpu_statistics = cpu.fitted_nodes[1]
    @test accelerated_statistics.means ≈ cpu_statistics.means rtol=1.0f-6 atol=1.0f-6
    @test accelerated_statistics.scales ≈ cpu_statistics.scales rtol=1.0f-5 atol=1.0f-5
    @test report(accelerated_statistics).details.deterministic_reductions
    @test report(accelerated).details.accelerator_fit_nodes == [1, 2]
    @test report(accelerated.cpu_graph).details.execution === :reactant_fit_skeleton
    accelerated_head = accelerated.cpu_graph.fitted_nodes[2]
    @test report(accelerated_head).backend === :reactant
    @test report(accelerated_head).details.sufficient_statistics_backend === :reactant
    @test report(accelerated_head).details.solver_backend === :reactant

    raw_model = Chain(RidgeRegression(lambda=0.75, fit_intercept=false))
    raw_cpu = fit(raw_model, X, y; weights=weights,
                  context=FitContext(numerics=numerics))
    raw_accelerated = fit(raw_model, X, y; weights=weights, context=FitContext(
        backend=ReactantBackend(device=:cpu), numerics=numerics))
    @test predict(raw_accelerated, X) ≈ predict(raw_cpu, X) rtol=2.0f-4 atol=2.0f-4
    @test report(raw_accelerated).details.accelerator_fit_nodes == [1]
    @test report(raw_accelerated.cpu_graph).details.execution === :reactant_fit_skeleton
    @test report(last(raw_accelerated.cpu_graph.fitted_nodes)).details.solver_backend === :reactant
end

@testset "Reactant mixed CPU and device regions" begin
    X = Float32[-3 1; -2 0; -1 2; 1 -1; 2 1; 3 0]
    y = [:low, :low, :low, :high, :high, :high]
    model = Chain(RobustScale(), Standardize(), LogisticRegression(lambda=0.5f0))
    cpu = fit(model, X, y)
    accelerated = fit(model, X, y; context=FitContext(
        backend=ReactantBackend(device=:cpu, fallback=:cpu)))

    @test report(accelerated).backend === :reactant
    @test predict_proba(accelerated, X) ≈ predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    @test predict(accelerated, X) == predict(cpu, X)
    mixed_report = report(accelerated)
    @test mixed_report.details.host_nodes == [1]
    @test mixed_report.details.accelerator_nodes == [2, 3]
    @test mixed_report.details.phase_placement.inference ==
          [(node_id=1, device=:cpu), (node_id=2, device=:reactant),
           (node_id=3, device=:reactant)]
    @test any(location -> location.location === :region_boundary &&
              location.from_node == 1 && location.to_node == 2,
              mixed_report.details.transfer_locations)
    primitives = mixed_report.details.reactant_capabilities.inference.graph.primitives
    @test any(node -> node.operation === :transfer_host_to_device, primitives)
    @test all(node -> node.device === :reactant,
              filter(node -> node.semantic_node_id >= 2, primitives))
    fitted_node_ids = objectid.(accelerated.cpu_graph.fitted_nodes)
    predict_proba(accelerated, reverse(X; dims=1))
    @test objectid.(accelerated.cpu_graph.fitted_nodes) == fitted_node_ids

    sandwich = Chain(Standardize(), RobustScale(),
                     LogisticRegression(lambda=0.5f0))
    sandwich_cpu = fit(sandwich, X, y)
    sandwich_accelerated = fit(sandwich, X, y; context=FitContext(
        backend=ReactantBackend(device=:cpu, fallback=:cpu)))
    @test predict_proba(sandwich_accelerated, X) ≈
          predict_proba(sandwich_cpu, X) rtol=1.0f-4 atol=1.0f-5
    sandwich_report = report(sandwich_accelerated)
    @test sandwich_report.details.accelerator_nodes == [1, 3]
    @test sandwich_report.details.host_nodes == [2]
    @test count(location -> location.location === :region_boundary,
                sandwich_report.details.transfer_locations) == 2
    sandwich_primitives = sandwich_report.details.reactant_capabilities.inference.graph.primitives
    @test any(node -> node.operation === :transfer_device_to_host, sandwich_primitives)
    @test any(node -> node.operation === :transfer_host_to_device, sandwich_primitives)

    clipped = Chain(MinMaxScale(feature_range=(-1.0f0, 1.0f0), clip=true),
                    RobustScale(), Standardize(),
                    LogisticRegression(lambda=0.5f0))
    clipped_cpu = fit(clipped, X, y)
    clipped_accelerated = fit(clipped, X, y; context=FitContext(
        backend=ReactantBackend(device=:cpu, fallback=:cpu)))
    extrapolated = Float32[-30 10; 30 -10]
    @test predict_proba(clipped_accelerated, extrapolated) ≈
          predict_proba(clipped_cpu, extrapolated) rtol=1.0f-4 atol=1.0f-5
    clipped_report = report(clipped_accelerated)
    @test clipped_report.details.accelerator_nodes == [1, 3, 4]
    @test clipped_report.details.host_nodes == [2]
    @test any(node -> node.operation === :clamp && node.device === :reactant,
              clipped_report.details.reactant_capabilities.inference.graph.primitives)
end

@testset "Reactant mixed branched DAG regions" begin
    X = Float32[-3 1; -2 0; -1 2; 1 -1; 2 1; 3 0]
    y = [:low, :low, :low, :high, :high, :high]
    model = Chain(Parallel(Standardize(), RobustScale()), Concatenate(),
                  LogisticRegression(lambda=0.5f0))
    cpu = fit(model, X, y)
    accelerated = fit(model, X, y; context=FitContext(
        backend=ReactantBackend(device=:cpu, fallback=:cpu)))

    @test predict_proba(accelerated, X) ≈
          predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    @test predict(accelerated, X) == predict(cpu, X)
    dag_report = report(accelerated)
    @test dag_report.backend === :reactant
    @test dag_report.details.accelerator_nodes == [1, 4]
    @test dag_report.details.host_nodes == [2, 3]
    @test any(transfer -> transfer.location === :region_boundary &&
              transfer.from_node == 1 && transfer.to_node == 3 &&
              transfer.direction === :device_to_host,
              dag_report.details.transfer_locations)
    @test any(transfer -> transfer.location === :region_boundary &&
              transfer.from_node == 3 && transfer.to_node == 4 &&
              transfer.direction === :host_to_device,
              dag_report.details.transfer_locations)
    primitives = dag_report.details.reactant_capabilities.inference.graph.primitives
    @test any(node -> node.operation === :transfer_device_to_host &&
              node.semantic_node_id == 3, primitives)
    @test any(node -> node.operation === :transfer_host_to_device &&
              node.semantic_node_id == 4, primitives)
end

@testset "Reactant numeric imputation lowering" begin
    X = Matrix{Union{Missing,Float32}}([
        1 10; missing 20; 3 missing; 4 40; 5 50; 6 60
    ])
    y = Float32[1, 2, 3, 4, 5, 6]
    model = Chain(Impute(strategy=:constant, fill_value=0.5f0), Standardize(),
                  RidgeRegression(lambda=0.1f0))
    cpu = fit(model, X, y)
    accelerated = fit(model, X, y;
        context=FitContext(backend=ReactantBackend(device=:cpu)))
    Xnew = Matrix{Union{Missing,Float32}}([missing 15; 2 missing; 7 70])
    @test predict(accelerated, Xnew) ≈ predict(cpu, Xnew) rtol=1.0f-4 atol=1.0f-5
    records = report(accelerated).details.reactant_capabilities.inference.primitives
    @test any(record -> record.operation === :missing_mask && record.supported, records)
    @test any(record -> record.operation === :select_fill && record.supported, records)
end

@testset "Reactant gather and concatenate graph lowering" begin
    X = Float32[1 10 0; 2 20 1; 3 30 0; 4 40 1; 5 50 0; 6 60 1]
    y = Float32[1, 2, 3, 4, 5, 6]
    model = Chain(Parallel(Standardize(), PCA(n_components=1)),
                  Concatenate(), RidgeRegression(lambda=0.1f0))
    cpu = fit(model, X, y)
    accelerated = fit(model, X, y;
        context=FitContext(backend=ReactantBackend(device=:cpu)))
    @test predict(accelerated, X) ≈ predict(cpu, X) rtol=1.0f-4 atol=1.0f-5
    operations = map(record -> record.operation,
                     report(accelerated).details.reactant_capabilities.inference.primitives)
    @test :concatenate in operations

    selected_model = Chain(Select(1, 3), Standardize(), LinearRegression())
    selected_cpu = fit(selected_model, X, y)
    selected_accelerated = fit(selected_model, X, y;
        context=FitContext(backend=ReactantBackend(device=:cpu)))
    @test predict(selected_accelerated, X) ≈ predict(selected_cpu, X) rtol=1.0f-4 atol=1.0f-5
    selected_operations = map(record -> record.operation,
        report(selected_accelerated).details.reactant_capabilities.inference.primitives)
    @test :gather in selected_operations
end

@testset "Reactant decomposition projection lowering" begin
    X = Float32[-3 1 0; -2 0 1; -1 2 -1; 1 -1 2; 2 1 1; 3 0 -2]
    regression_target = Float32[-5, -4, -1, 1, 5, 6]
    classes = [:low, :low, :low, :high, :high, :high]
    context = FitContext(backend=ReactantBackend(device=:cpu))

    models_and_targets = (
        (Chain(PCA(n_components=2), LinearRegression()), regression_target),
        (Chain(PCA(n_components=2, whiten=true), LogisticRegression(lambda=0.5f0)), classes),
        (Chain(TruncatedSVD(n_components=2), RidgeRegression(lambda=0.25f0)),
         regression_target),
    )
    for (model, target) in models_and_targets
        cpu = fit(model, X, target)
        accelerated = fit(model, X, target; context=context)
        @test predict(accelerated, X) == predict(cpu, X) ||
              isapprox(predict(accelerated, X), predict(cpu, X); rtol=1.0f-4, atol=1.0f-5)
        if target === classes
            @test predict_proba(accelerated, X) ≈
                  predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
        end
        operations = map(record -> record.operation,
                         report(accelerated).details.reactant_capabilities.inference.primitives)
        @test :matmul in operations
    end
end

@testset "Reactant affine lowering registry and prediction heads" begin
    X = Float32[-3 1; -2 0; -1 2; 1 -1; 2 1; 3 0]
    regression_target = Float32[-5, -4, -1, 1, 5, 6]
    context = FitContext(backend=ReactantBackend(device=:cpu))

    for model in (Chain(LinearRegression()),
                  Chain(Standardize(), RidgeRegression(lambda=0.25f0)),
                  Chain(MinMaxScale(feature_range=(-1.0f0, 1.0f0)),
                        LinearRegression()))
        cpu = fit(model, X, regression_target)
        accelerated = fit(model, X, regression_target; context=context)
        @test predict(accelerated, X) ≈ predict(cpu, X) rtol=1.0f-4 atol=1.0f-5
        device_prediction = predict(accelerated, Reactant.to_rarray(X); output=:device)
        @test device_prediction isa Reactant.AbstractConcreteArray
        @test Array(device_prediction) ≈ predict(cpu, X) rtol=1.0f-4 atol=1.0f-5
        @test_throws Tilia.UnsupportedDataError predict_proba(accelerated, X)
        @test !report(accelerated).details.accelerated_logistic_objective
    end

    classes = [:low, :low, :low, :high, :high, :high]
    classifier = Chain(MinMaxScale(feature_range=(-2.0f0, 2.0f0)),
                       LogisticRegression(lambda=0.5f0))
    cpu_classifier = fit(classifier, X, classes)
    accelerated_classifier = fit(classifier, X, classes; context=context)
    @test predict_proba(accelerated_classifier, X) ≈
          predict_proba(cpu_classifier, X) rtol=1.0f-4 atol=1.0f-5
    primitive_records = report(accelerated_classifier).details.reactant_capabilities.predict_proba.primitives
    @test any(record -> record.operation === :affine && record.supported,
              primitive_records)

    clipped = Chain(MinMaxScale(feature_range=(-1.0f0, 1.0f0), clip=true),
                    LinearRegression())
    clipped_cpu = fit(clipped, X, regression_target)
    clipped_accelerated = fit(clipped, X, regression_target; context=context)
    outside_training_range = Float32[-8 4; 0 0; 8 -4]
    @test predict(clipped_accelerated, outside_training_range) ≈
          predict(clipped_cpu, outside_training_range) rtol=1.0f-4 atol=1.0f-5
    @test any(record -> record.operation === :clamp && record.supported,
              report(clipped_accelerated).details.reactant_capabilities.inference.primitives)
end

@testset "Reactant explicit GPU or compilation failure contract" begin
    X = Float32[-2 0; -1 1; 1 -1; 2 0]
    y = [:negative, :negative, :positive, :positive]
    model = Chain(Standardize(), LogisticRegression(lambda=1.0f0))
    previous_backend = Reactant.XLA.default_backend()
    previous_platform = Reactant.XLA.platform_name(previous_backend)

    fitted = fit(model, X, y; context=FitContext(
        backend=ReactantBackend(device=:gpu, fallback=:cpu)))
    fitted_report = report(fitted)
    if fitted_report.backend === :reactant
        @test fitted_report.details.device in (:cuda, :rocm, :metal)
        @test predict_proba(fitted, X) ≈ predict_proba(fit(model, X, y), X) rtol=1.0f-4 atol=1.0f-5
    else
        @test fitted_report.backend === :cpu
        @test fitted_report.details.requested_backend === :reactant
        @test any(reason -> occursin("compilation failed", lowercase(reason)),
                  fitted_report.details.unsupported_operations)
    end
    @test Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == previous_platform
    @test Reactant.XLA.default_backend() === previous_backend
end

@testset "Reactant expanded numerical coverage" begin
    X64 = Float64[-3 0; -2 1; -1 -1; 1 1; 2 -1; 3 0]
    y3 = [:left, :left, :middle, :middle, :right, :right]
    weights = Float64[1, 2, 1, 3, 2, 1]
    model = Chain(Standardize(), LogisticRegression(lambda=0.5))
    cpu = fit(model, X64, y3; weights=weights)
    accelerated = fit(model, X64, y3; weights=weights,
        context=FitContext(backend=ReactantBackend(device=:cpu)))

    @test predict_proba(accelerated, X64) ≈ predict_proba(cpu, X64) rtol=1e-8 atol=1e-10
    @test predict(accelerated, X64) == predict(cpu, X64)
    @test report(accelerated).details.accelerator_fit_nodes == [1, 2]
    @test report(last(accelerated.cpu_graph.fitted_nodes)).details.optimizer_backend === :reactant
    @test length(report(last(accelerated.cpu_graph.fitted_nodes)).details.convergence) == 3
    accelerated_histories = report(last(accelerated.cpu_graph.fitted_nodes)).details.objective_history
    cpu_histories = report(last(cpu.fitted_nodes)).details.objective_history
    @test all(isapprox(accelerated_histories[index], cpu_histories[index];
                       rtol=1e-8, atol=1e-10) for index in eachindex(cpu_histories))
    cpu_objective = sum(last(history) for history in
                        report(last(cpu.fitted_nodes)).details.objective_history)
    @test report(accelerated).details.accelerated_objective_value ≈ cpu_objective rtol=1e-8

    @test_throws Tilia.UnsupportedDataError predict_proba(accelerated, zeros(Float64, 0, 2))
    @test_throws Tilia.SchemaMismatchError predict_proba(accelerated, zeros(Float64, 2, 3))

    large = repeat(X64, 350, 1)
    @test predict_proba(accelerated, large) ≈ predict_proba(cpu, large) rtol=1e-8 atol=1e-10

    concurrent_inputs = [X64, reverse(X64; dims=1), X64[1:3, :], large]
    tasks = map(concurrent_inputs) do input
        Threads.@spawn predict_proba(accelerated, input)
    end
    concurrent_results = fetch.(tasks)
    @test all(zip(concurrent_results, concurrent_inputs)) do (result, input)
        isapprox(result, predict_proba(cpu, input); rtol=1e-8, atol=1e-10)
    end

    fill!(last(accelerated.cpu_graph.fitted_nodes).coefficients, 0)
    fill!(last(accelerated.cpu_graph.fitted_nodes).intercept, 0)
    fill!(last(cpu.fitted_nodes).coefficients, 0)
    fill!(last(cpu.fitted_nodes).intercept, 0)
    @test predict(accelerated, X64) == predict(cpu, X64) == fill(:left, size(X64, 1))
end

@testset "Reactant numerical primitive capabilities" begin
    X = Float32[-2 0; -1 1; 1 -1; 2 0]
    y = [:negative, :negative, :positive, :positive]
    context = FitContext(backend=ReactantBackend(device=:cpu))

    standalone = Chain(LogisticRegression(lambda=1.0f0))
    cpu = fit(standalone, X, y)
    accelerated = fit(standalone, X, y; context=context)
    @test predict_proba(accelerated, X) ≈ predict_proba(cpu, X) rtol=1.0f-4 atol=1.0f-5
    capability_report = report(accelerated).details.reactant_capabilities
    @test all(record -> record.supported, capability_report.predict_proba.primitives)
    @test all(record -> record.phase === :inference,
              capability_report.predict_proba.primitives)
    @test any(record -> !record.supported && record.phase === :fit,
              capability_report.fit.primitives)

    unsupported = Chain(Standardize(), MeanRegressor())
    error = try
        fit(unsupported, X, Float32[1, 2, 3, 4]; context=context)
        nothing
    catch caught
        caught
    end
    @test error isa Tilia.UnsupportedBackendError
    @test occursin("fill", sprint(showerror, error))
    @test occursin("inference", sprint(showerror, error))
end

@testset "Reactant compilation cache does not retain fitted parameters" begin
    X = Float32[-2 0; -1 1; 1 -1; 2 0]
    y_first = [:negative, :negative, :positive, :positive]
    y_second = [:positive, :positive, :negative, :negative]
    model = Chain(Standardize(), LogisticRegression(lambda=1.0f0))
    cache = CompilationCache()
    context = FitContext(backend=ReactantBackend(device=:cpu), cache=cache)

    cpu_first = fit(model, X, y_first)
    cpu_second = fit(model, X, y_second)
    accelerated_first = fit(model, X, y_first; context=context)
    accelerated_second = fit(model, X, y_second; context=context)

    @test predict_proba(accelerated_first, X) ≈ predict_proba(cpu_first, X) rtol=1.0f-4 atol=1.0f-5
    @test predict_proba(accelerated_second, X) ≈ predict_proba(cpu_second, X) rtol=1.0f-4 atol=1.0f-5
    @test predict_proba(accelerated_first, X) != predict_proba(accelerated_second, X)
    @test report(accelerated_second).details.compilation_cache_hits >= 1

    accelerated_first.cpu_graph.fitted_nodes[1].means[1] += 0.5f0
    accelerated_first.cpu_graph.fitted_nodes[2].intercept[1] += 0.25f0
    @test predict_proba(accelerated_first, X) ≈
          predict_proba(accelerated_first.cpu_graph, X) rtol=1.0f-4 atol=1.0f-5
end

@testset "Reactant concurrent fit isolation and backend restoration" begin
    X = Float64[-2 1; -1 2; 1 -1; 2 -2; 3 1; 4 3]
    targets = (Float64[-4, -2, 1, 3, 5, 8],
               Float64[7, 5, 2, 0, -3, -6])
    model = Chain(RidgeRegression(lambda=0.5, fit_intercept=false))
    cache = CompilationCache()
    context = FitContext(backend=ReactantBackend(device=:cpu), cache=cache)
    previous_backend = Reactant.XLA.default_backend()

    tasks = map(targets) do target
        Threads.@spawn fit(model, X, target; context=context)
    end
    accelerated = fetch.(tasks)
    references = map(target -> fit(model, X, target), targets)

    @test all(eachindex(targets)) do index
        isapprox(predict(accelerated[index], X), predict(references[index], X);
                 rtol=1e-10, atol=1e-12)
    end
    @test predict(accelerated[1], X) != predict(accelerated[2], X)
    @test sum(report(fitted).details.compilation_cache_hits for fitted in accelerated) >= 1
    @test all(fitted -> report(fitted.cpu_graph).details.execution ===
                        :reactant_fit_skeleton, accelerated)
    @test Reactant.XLA.default_backend() === previous_backend
end

@testset "Reactant portable persistence excludes compiled state" begin
    X = Float64[-2 1; -1 2; 1 -1; 2 -2; 3 1; 4 3]
    y = Float64[-4, -2, 1, 3, 5, 8]
    model = Chain(RidgeRegression(lambda=0.5, fit_intercept=false))
    cache = CompilationCache()
    accelerated = fit(model, X, y; context=FitContext(
        backend=ReactantBackend(device=:cpu), cache=cache))
    expected = predict(accelerated, X)
    cache_size_before = report(accelerated).details.compilation_cache_size

    mktempdir() do directory
        save_model(directory, accelerated)
        loaded = load_model(directory)

        @test loaded isa Tilia.FittedGraph
        @test !hasproperty(loaded, :cache)
        @test report(loaded).backend === :cpu
        @test predict(loaded, X) ≈ expected rtol=1e-12 atol=1e-12
        @test last(loaded.fitted_nodes).coefficients ≈
              last(accelerated.cpu_graph.fitted_nodes).coefficients
        @test report(accelerated).details.compilation_cache_size == cache_size_before
        empty!(cache)
        @test predict(loaded, X) ≈ expected rtol=1e-12 atol=1e-12
    end
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
    @test fallback_report.details.requested_backend === :reactant
    @test all(item -> item.device === :cpu,
              fallback_report.details.phase_placement.inference)
    @test fallback_report.details.transfer_accounting.estimated_bytes == 0
    @test predict(fallback, X) == fill(2.0f0, 3)
end
