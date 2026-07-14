module TiliaDifferentiationExt

using Tilia
using DifferentiationInterface

struct DifferentiatedObjective{F,B} <: Tilia.AbstractObjective
    function_value::F
    backend::B
end

Tilia.autodiff_objective(function_value, backend) =
    DifferentiatedObjective(function_value, backend)

Tilia.value(objective::DifferentiatedObjective, parameters, data) =
    objective.function_value(parameters, data)

function Tilia.gradient!(destination, objective::DifferentiatedObjective,
                         parameters, data)
    differentiated = parameters_value -> objective.function_value(parameters_value, data)
    DifferentiationInterface.gradient!(differentiated, destination,
                                       objective.backend, parameters)
end

function Tilia.value_and_gradient!(destination, objective::DifferentiatedObjective,
                                   parameters, data)
    differentiated = parameters_value -> objective.function_value(parameters_value, data)
    DifferentiationInterface.value_and_gradient!(differentiated, destination,
                                                 objective.backend, parameters)
end

end
