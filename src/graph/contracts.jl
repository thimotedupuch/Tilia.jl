# Feature-count and backend declarations live after estimator definitions so
# graph metadata remains centralized without introducing backend checks in
# individual fitting implementations.
preserves_feature_count(::Union{Standardize,Impute}) = true

backend_compatibility(::Union{Standardize,LogisticRegression}) = (:cpu, :reactant)
