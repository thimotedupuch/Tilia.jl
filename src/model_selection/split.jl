"""Deterministic K-fold splitter; fold sizes differ by at most one observation."""
struct KFold
    n_splits::Int
    shuffle::Bool
    seed::UInt64
    function KFold(n_splits::Integer=5; shuffle::Bool=false, seed::Integer=0)
        n_splits >= 2 || throw(InvalidHyperparameterError("KFold n_splits must be at least 2."))
        seed >= 0 || throw(InvalidHyperparameterError("KFold seed must be nonnegative."))
        new(Int(n_splits), shuffle, UInt64(seed))
    end
end

"""Return `(train_indices, test_indices)` for every K-fold split."""
function split(splitter::KFold, observations::Integer)
    observations >= splitter.n_splits || throw(UnsupportedDataError(
        "KFold n_splits=$(splitter.n_splits) exceeds $observations observations."))
    indices = collect(1:observations)
    splitter.shuffle && Random.shuffle!(Random.Xoshiro(splitter.seed), indices)
    base, remainder = divrem(observations, splitter.n_splits)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    start = 1
    for fold in 1:splitter.n_splits
        fold_size = base + (fold <= remainder)
        stop = start + fold_size - 1
        test_indices = sort(indices[start:stop])
        test_set = Set(test_indices)
        train_indices = [index for index in 1:observations if index ∉ test_set]
        push!(folds, (train_indices, test_indices))
        start = stop + 1
    end
    folds
end

function _test_count(observations, test_size)
    count = test_size isa Integer ? Int(test_size) : ceil(Int, observations * test_size)
    1 <= count < observations || throw(InvalidHyperparameterError(
        "test_size must select between 1 and $(observations - 1) observations."))
    count
end

"""
Split arrays into deterministic train and test partitions. Supplying `stratify`
preserves every class in both partitions when each class has at least two rows.
"""
function train_test_split(X::Union{AbstractMatrix,ColumnTable}, y::AbstractVector; test_size=0.25,
                          shuffle::Bool=true, seed::Integer=0, stratify=nothing)
    size(X, 1) == length(y) || throw(SchemaMismatchError("features and target observation counts must agree."))
    observations = length(y)
    test_count = _test_count(observations, test_size)
    rng = Random.Xoshiro(seed)
    test_indices = if stratify === nothing
        indices = collect(1:observations)
        shuffle && Random.shuffle!(rng, indices)
        sort(indices[1:test_count])
    else
        length(stratify) == observations || throw(SchemaMismatchError("stratify and target lengths must agree."))
        classes = sort!(unique(stratify))
        groups = [findall(==(class), stratify) for class in classes]
        all(length(group) >= 2 for group in groups) || throw(UnsupportedDataError(
            "stratified splitting requires at least two observations per class."))
        shuffle && foreach(group -> Random.shuffle!(rng, group), groups)
        raw = [test_count * length(group) / observations for group in groups]
        counts = clamp.(floor.(Int, raw), 1, length.(groups) .- 1)
        while sum(counts) < test_count
            candidates = findall(index -> counts[index] < length(groups[index]) - 1, eachindex(groups))
            isempty(candidates) && throw(UnsupportedDataError("requested stratified test size cannot preserve all classes."))
            best = candidates[argmax([raw[index] - counts[index] for index in candidates])]
            counts[best] += 1
        end
        while sum(counts) > test_count
            candidates = findall(index -> counts[index] > 1, eachindex(groups))
            isempty(candidates) && throw(UnsupportedDataError("requested stratified test size is too small for all classes."))
            best = candidates[argmax([counts[index] - raw[index] for index in candidates])]
            counts[best] -= 1
        end
        sort!(reduce(vcat, group[1:count] for (group, count) in zip(groups, counts)))
    end
    test_set = Set(test_indices)
    train_indices = [index for index in 1:observations if index ∉ test_set]
    (select_rows(X, train_indices), select_rows(X, test_indices), y[train_indices], y[test_indices],
     train_indices, test_indices)
end
