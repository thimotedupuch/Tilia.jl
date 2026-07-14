"""Internal numerical solvers shared by statistical models."""
module Solvers

using LinearAlgebra
using Random

include("least_squares.jl")
include("newton.jl")
include("coordinate_descent.jl")
include("iterative.jl")

export LeastSquaresResult, least_squares, ridge_least_squares
export NewtonResult, binary_logistic_newton, binary_logistic_irls
export CoordinateDescentResult, elastic_net_coordinate_descent
export IterativeResult, conjugate_gradient, lsqr, lbfgs, newton_cg
export proximal_gradient, fista, stochastic_gradient, expectation_maximization

end
