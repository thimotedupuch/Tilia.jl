# Numerical behavior

`FitContext` owns backend, random stream, numerical policy, determinism, and
compilation cache. Float32 and Float64 behavior is tested across model families.
Stable sigmoid, log-sum-exp, log-softmax, binary loss, norms, weighted
statistics, distance, ranking, covariance, and sparse kernels are centralized.

Iterative reports expose objective histories, iteration counts, convergence,
and warnings. Numerical failures use typed errors rather than silent NaNs.
