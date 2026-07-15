@testset "Graph optimization and tracing" begin
    first_transform = Tilia.TransformNode(1, Standardize())
    redundant = Tilia.ConversionNode(2, :dense, :dense)
    dead_transform = Tilia.TransformNode(3, Standardize())
    predictor = Tilia.PredictorNode(4, MeanRegressor())
    graph = Tilia.SemanticGraph(Tilia.AbstractGraphNode[
        first_transform, redundant, dead_transform, predictor],
        [(1, 2), (2, 3), (1, 4)])
    @test Tilia.validate_graph(graph) === graph

    conversions_removed = Tilia.redundant_conversion_elimination(graph)
    @test length(conversions_removed.nodes) == 3
    @test all(!(node isa Tilia.ConversionNode) for node in conversions_removed.nodes)
    @test (1, 3) in conversions_removed.edges

    optimized_graph = Tilia.optimize(graph)
    @test length(optimized_graph.nodes) == 2
    @test optimized_graph.edges == [(1, 2)]
    @test optimized_graph.nodes[end] isa Tilia.PredictorNode

    plan = Tilia.execution_plan(conversions_removed)
    @test plan.order == [1, 2, 3]
    @test plan.peak_buffers == 2 # dead branch output is released before the predictor
    @test length(plan.buffers) == 3

    visualization = Tilia.graph_data(optimized_graph)
    @test length(visualization.nodes) == 2
    @test visualization.edges == [(1, 2)]
    @test visualization.nodes[1].learns
    @test visualization.nodes[2].consumes_target

    transform_contract = Tilia.node_contract(first_transform)
    @test transform_contract.input.rows_are_observations
    @test transform_contract.output_schema_rule == :model_dispatch
    @test transform_contract.learns_state
    @test !transform_contract.consumes_target
    @test !transform_contract.changes_row_count
    @test !transform_contract.changes_feature_count
    @test transform_contract.valid_at_inference
    @test !transform_contract.sparse_compatible
    @test !transform_contract.missing_compatible
    @test transform_contract.backend_compatibility == (:cpu, :reactant)

    predictor_contract = Tilia.node_contract(predictor)
    @test predictor_contract.consumes_target
    @test predictor_contract.changes_feature_count
    @test predictor_contract.backend_compatibility == (:cpu,)
    @test Tilia.validate_backend(graph, :cpu) === graph
    @test_throws Tilia.UnsupportedBackendError Tilia.validate_backend(graph, :reactant)

    reactant_graph = Tilia.build_graph(Chain(Standardize(), LogisticRegression()))
    @test Tilia.validate_backend(reactant_graph, :reactant) === reactant_graph

    conversion_contract = Tilia.node_contract(redundant)
    @test !conversion_contract.learns_state
    @test conversion_contract.output_schema_rule == :representation_conversion

    X = [1.0 10.0; 2.0 12.0; 4.0 18.0; 8.0 30.0]
    y = [2.0, 4.0, 9.0, 17.0]
    pipeline = Chain(Standardize(center=true, scale=false),
                     Standardize(center=false, scale=true),
                     LinearRegression())
    fitted = fit(pipeline, X, y)
    optimized = Tilia.optimize(fitted)
    @test length(fitted.fitted_nodes) == 3
    @test length(optimized.fitted_nodes) == 2
    @test optimized.fitted_nodes[1] isa Tilia.FittedStandardize
    @test predict(optimized, X) ≈ predict(fitted, X) atol=1e-12
    @test Tilia.transform(optimized.fitted_nodes[1], X) ≈
          Tilia.transform(fitted.fitted_nodes[2],
                          Tilia.transform(fitted.fitted_nodes[1], X)) atol=1e-12
    @test report(optimized).details.optimization.fused_transforms == 1
    @test report(optimized).details.optimization.original_nodes == 3
    @test length(report(optimized).details.fit_execution_graph.nodes) == 2
    @test length(report(optimized).details.inference_execution_graph.nodes) == 2
    @test length(report(fitted).details.node_timings) == 3
    @test all(timing.nanoseconds >= 0 for timing in report(fitted).details.node_timings)
    fit_execution = report(fitted).details.fit_execution_graph
    inference_execution = report(fitted).details.inference_execution_graph
    @test fit_execution isa Tilia.NumericalExecutionGraph
    @test inference_execution isa Tilia.NumericalExecutionGraph
    @test fit_execution.phase == :fit
    @test inference_execution.phase == :inference
    @test [node.operation for node in fit_execution.nodes] ==
          [:fit_transform, :fit_transform, :fit]
    @test [node.operation for node in inference_execution.nodes] ==
          [:transform, :transform, :predict]
    @test all(node -> node.device == :cpu, fit_execution.nodes)
    @test all(node -> node.mutability == :owned_output, fit_execution.nodes)
    @test fit_execution.nodes[1].input_shape == size(X)
    @test fit_execution.nodes[1].output_shape == size(X)
    @test fit_execution.nodes[end].input_shape == size(X)
    @test fit_execution.nodes[1].element_type == Float64
    @test fit_execution.nodes[1].representation == :dense
    @test fit_execution.peak_buffers == Tilia.execution_plan(fitted.graph).peak_buffers

    execution = Tilia.trace(optimized, X)
    @test execution.output ≈ predict(optimized, X)
    @test length(execution.nodes) == 2
    @test execution.nodes[1].operation == :transform
    @test execution.nodes[2].operation == :predict
    @test execution.nodes[1].input_shape == size(X)
    @test execution.nodes[2].output_shape == (size(X, 1),)
    @test execution.total_nanoseconds >= sum(record.nanoseconds for record in execution.nodes)

    classifier = fit(Chain(Standardize(), LogisticRegression()),
        [-2.0 0.0; -1.0 1.0; 1.0 -1.0; 2.0 0.0], [:n, :n, :p, :p])
    lowered_probability = Tilia._execute_inference_graph(classifier,
        [-2.0 0.0; 2.0 0.0], :predict_proba)
    @test lowered_probability.graph.nodes[end].operation == :predict_proba
    @test lowered_probability.graph.nodes[end].output_shape == (2, 2)
    @test vec(sum(lowered_probability.output; dims=2)) ≈ ones(2)
    probability_trace = Tilia.trace(classifier,
        [-2.0 0.0; 2.0 0.0]; operation=:predict_proba)
    @test size(probability_trace.output) == (2, 2)
    @test vec(sum(probability_trace.output; dims=2)) ≈ ones(2)
    @test probability_trace.nodes[end].operation == :predict_proba

    triple = fit(Chain(Standardize(center=true, scale=false),
                       Standardize(center=false, scale=true),
                       Standardize(), MeanRegressor()), X, y)
    triple_optimized = Tilia.optimize(triple)
    @test length(triple_optimized.fitted_nodes) == 2
    @test report(triple_optimized).details.optimization.fused_transforms == 2
    @test predict(triple_optimized, X) ≈ predict(triple, X)

    constant_graph = Tilia.SemanticGraph(Tilia.AbstractGraphNode[
        Tilia.ConstantNode(1, 2.0), Tilia.ConstantNode(2, 3.0),
        Tilia.BinaryOperationNode(3, :multiply, 1, 2)], [(1, 3), (2, 3)])
    folded = Tilia.constant_folding(constant_graph)
    @test folded.nodes[3] isa Tilia.ConstantNode
    @test folded.nodes[3].value == 6.0
    @test isempty(folded.edges)
    folded_optimized = Tilia.optimize(constant_graph)
    @test length(folded_optimized.nodes) == 1
    @test only(folded_optimized.nodes).value == 6.0

    placement = Tilia.device_placement(fitted.graph;
        default=:cpu, overrides=Dict(2 => :reactant, 3 => :reactant))
    @test [assignment.device for assignment in placement.assignments] ==
          [:cpu, :reactant, :reactant]
    @test length(placement.transfers) == 1
    @test only(placement.transfers).from_device == :cpu
    @test only(placement.transfers).to_device == :reactant
    duplicate = only(placement.transfers)
    @test length(Tilia.coalesce_transfers([duplicate, duplicate])) == 1
    @test_throws Tilia.GraphValidationError Tilia.device_placement(fitted.graph;
        overrides=Dict(99 => :reactant))
end
