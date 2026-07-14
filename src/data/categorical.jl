"""
Tilia-owned categorical representation. Code zero denotes a missing or unknown
level according to the column policy; positive codes index `pool`.
"""
struct CategoricalColumn{T,C<:AbstractVector{<:Integer},P<:AbstractVector{T}} <: AbstractVector{Union{Missing,T}}
    codes::C
    pool::P
    ordered::Bool
    unknown_policy::Symbol
    function CategoricalColumn(codes::C, pool::P; ordered::Bool=false,
                               unknown_policy::Symbol=:error) where {C<:AbstractVector{<:Integer},P<:AbstractVector}
        unknown_policy in (:error, :ignore) || throw(InvalidHyperparameterError(
            "categorical unknown_policy must be :error or :ignore."))
        all(code -> 0 <= code <= length(pool), codes) || throw(ArgumentError(
            "categorical codes must lie between zero and the pool length."))
        length(unique(pool)) == length(pool) || throw(ArgumentError("categorical pool levels must be unique."))
        new{eltype(P),C,P}(codes, pool, ordered, unknown_policy)
    end
end

Base.length(column::CategoricalColumn) = length(column.codes)
Base.size(column::CategoricalColumn) = (length(column),)
Base.IndexStyle(::Type{<:CategoricalColumn}) = IndexLinear()
Base.getindex(column::CategoricalColumn, index::Integer) =
    iszero(column.codes[index]) ? missing : column.pool[column.codes[index]]
Base.getindex(column::CategoricalColumn, indices::AbstractVector{<:Integer}) =
    CategoricalColumn(column.codes[indices], copy(column.pool);
                      ordered=column.ordered, unknown_policy=column.unknown_policy)

function categorical_column(values; ordered::Bool=false, unknown_policy::Symbol=:error)
    observed = collect(skipmissing(values))
    pool = try
        sort!(unique(observed))
    catch
        unique(observed)
    end
    lookup = Dict(level => index for (index, level) in enumerate(pool))
    codes = [ismissing(value) ? 0 : lookup[value] for value in values]
    CategoricalColumn(codes, pool; ordered=ordered, unknown_policy=unknown_policy)
end
