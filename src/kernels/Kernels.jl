"""Shared, model-independent numerical primitives used throughout Tilia."""
module Kernels

using LinearAlgebra
using SparseArrays
using Statistics

include("reductions.jl")
include("logexp.jl")
include("normalization.jl")
include("distances.jl")
include("losses.jl")
include("sparse.jl")
include("kernel_functions.jl")
include("statistics.jl")
include("selection.jl")

export stable_sum, weighted_sum, weighted_mean, weighted_variance, stable_norm
export reduction_sum, reduction_mean, extrema_values, argmin_index, argmax_index
export logsumexp, softmax, logsoftmax, sigmoid, binary_cross_entropy
export clip_values
export normalize_rows, squared_euclidean, euclidean, manhattan, cosine_distance
export pairwise_distances, pairwise_distance_blocks
export mean_squared_error, root_mean_squared_error, log_loss
export scale_columns, scale_columns!
export gram_matrix
export covariance_matrix, weighted_covariance, contingency_matrix, class_counts, histogram_counts
export topk_indices, quantile_value, rank_values, mahalanobis_distance
export sparse_column_sums, sparse_dot, sparse_matvec, center_sparse

end
