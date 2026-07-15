using Random
using Statistics
using Tilia

let rng = Xoshiro(42)
for observations in (100, 1_000, 10_000)
    source = (
        value=Union{Missing,Float64}[
            rand(rng) < 0.05 ? missing : randn(rng) for _ in 1:observations
        ],
        category=[rand(rng, (:a, :b, :c, :d)) for _ in 1:observations],
    )
    table = column_table(source)
    imputer = fit(Impute(), table)
    imputed = transform(imputer, table)
    encoder = fit(OneHotEncode(), imputed)
    impute_operation = () -> transform(imputer, table)
    encode_operation = () -> transform(encoder, imputed)
    impute_first = @elapsed impute_operation()
    encode_first = @elapsed encode_operation()
    println((benchmark=:tabular_preprocessing, observations,
             impute_first_call_seconds=impute_first,
             impute_steady_state_seconds=median([@elapsed impute_operation() for _ in 1:3]),
             encode_first_call_seconds=encode_first,
             encode_steady_state_seconds=median([@elapsed encode_operation() for _ in 1:3]),
             output_features=size(encode_operation(), 2)))
end
end
