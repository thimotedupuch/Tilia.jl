struct ProjectionData{T,G}
    scores::Matrix{T}
    groups::G
    components::Tuple{Int,Int}
    variance_ratio::Union{Nothing,Vector{T}}
    title::String
end

struct ScreeData{T}
    explained::Vector{T}
    cumulative::Vector{T}
    threshold::T
end

struct BiplotData{T,G}
    scores::Matrix{T}
    loadings::Matrix{T}
    groups::G
    feature_names::Vector{String}
    components::Tuple{Int,Int}
    variance_ratio::Vector{T}
end

struct LoadingsData{T}
    values::Vector{T}
    names::Vector{String}
    component::Int
end

struct ComponentData{T}
    montage::Matrix{T}
    centers::Vector{Tuple{Float64,Float64,String}}
    colorrange::Tuple{T,T}
    title::String
end

struct ReconstructionData{T}
    montage::Matrix{T}
    row_labels::Vector{Tuple{Float64,Float64,String}}
    colorrange::Tuple{T,T}
end

_component_label(component, ratios) = ratios === nothing ? "Component $component" :
    "PC$component ($(round(100 * ratios[component]; digits=1))%)"

function _group_colors(groups)
    groups === nothing && return (:steelblue, nothing)
    levels = sort(unique(groups))
    palette = Makie.wong_colors()
    lookup = Dict(level => palette[mod1(index, length(palette))]
                  for (index, level) in enumerate(levels))
    return ([lookup[group] for group in groups], (levels, lookup))
end

function projectionplot(fitted, X::AbstractMatrix; groups=nothing,
                        components::Tuple{Int,Int}=(1, 2),
                        title=string(nameof(typeof(fitted.model))), kwargs...)
    scores = Tilia.transform(fitted, X)
    maximum(components) <= size(scores, 2) || throw(ArgumentError(
        "requested projection component exceeds the fitted component count"))
    groups === nothing || length(groups) == size(scores, 1) || throw(DimensionMismatch(
        "projection groups must match the observation count"))
    ratios = hasproperty(fitted, :explained_variance_ratio) ?
        collect(fitted.explained_variance_ratio) : nothing
    data = ProjectionData(Matrix(scores[:, collect(components)]), groups, components,
                          ratios, String(title))
    return projectionplot(data; kwargs...)
end

function screeplot(fitted::Tilia.FittedDecomposition; threshold::Real=0.95, kwargs...)
    hasproperty(fitted, :explained_variance_ratio) || throw(ArgumentError(
        "screeplot requires explained variance ratios"))
    0 < threshold <= 1 || throw(ArgumentError("threshold must lie in (0, 1]"))
    explained = collect(fitted.explained_variance_ratio)
    data = ScreeData(explained, cumsum(explained), convert(eltype(explained), threshold))
    return screeplot(data; kwargs...)
end


function biplot(fitted, X::AbstractMatrix; groups=nothing,
                feature_names=nothing, components::Tuple{Int,Int}=(1, 2), kwargs...)
    hasproperty(fitted, :components) && hasproperty(fitted, :explained_variance_ratio) ||
        throw(ArgumentError("biplot requires a fitted PCA or truncated-SVD model"))
    scores = Tilia.transform(fitted, X)
    maximum(components) <= size(scores, 2) || throw(ArgumentError(
        "requested biplot component exceeds the fitted component count"))
    groups === nothing || length(groups) == size(scores, 1) || throw(DimensionMismatch(
        "biplot groups must match the observation count"))
    names = feature_names === nothing ?
        ["x$index" for index in axes(fitted.components, 1)] : string.(feature_names)
    length(names) == size(fitted.components, 1) || throw(DimensionMismatch(
        "feature_names must match the fitted feature count"))
    selected = collect(components)
    data = BiplotData(Matrix(scores[:, selected]), Matrix(fitted.components[:, selected]),
                      groups, names, components, collect(fitted.explained_variance_ratio))
    return biplot(data; kwargs...)
end

function loadingsplot(fitted::Tilia.FittedDecomposition; component::Integer=1, feature_names=nothing,
                      max_features::Integer=20, kwargs...)
    hasproperty(fitted, :components) || throw(ArgumentError(
        "loadingsplot requires fitted component loadings"))
    1 <= component <= size(fitted.components, 2) || throw(ArgumentError(
        "component is outside the fitted component range"))
    names = feature_names === nothing ?
        ["x$index" for index in axes(fitted.components, 1)] : string.(feature_names)
    length(names) == size(fitted.components, 1) || throw(DimensionMismatch(
        "feature_names must match the fitted feature count"))
    values = collect(fitted.components[:, component])
    count = min(Int(max_features), length(values))
    selected = sort(sortperm(abs.(values); rev=true)[1:count]; by=index -> values[index])
    return loadingsplot(LoadingsData(values[selected], names[selected], Int(component)); kwargs...)
end

_component_matrix(fitted::Tilia.FittedDecomposition) = transpose(fitted.components)
_component_matrix(fitted::Tilia.FittedNMF) = fitted.components
_component_matrix(fitted::Tilia.FittedFastICA) = fitted.unmixing
_component_matrix(fitted::Tilia.FittedRandomProjection) = fitted.projection

function _tile_montage(images, rows, columns; gap=1)
    height, width, count = size(images)
    montage = fill(convert(eltype(images), NaN),
                   rows * height + (rows - 1) * gap,
                   columns * width + (columns - 1) * gap)
    centers = Tuple{Float64,Float64,String}[]
    for index in 1:count
        row, column = fldmod1(index, columns)
        y = (row - 1) * (height + gap) + 1
        x = (column - 1) * (width + gap) + 1
        montage[y:y + height - 1, x:x + width - 1] .= images[:, :, index]
        push!(centers, (x + (width - 1) / 2, y + height - 1, "C$index"))
    end
    montage, centers
end

function componentplot(fitted::Union{Tilia.FittedDecomposition,Tilia.FittedNMF,
                                     Tilia.FittedFastICA,Tilia.FittedRandomProjection};
                       shape::Tuple{Int,Int}, components=nothing,
                       columns::Integer=4, title="Learned components", kwargs...)
    matrix = Matrix(_component_matrix(fitted))
    prod(shape) == size(matrix, 2) || throw(DimensionMismatch(
        "component shape does not match the fitted feature count"))
    selected = components === nothing ? collect(axes(matrix, 1)) : collect(components)
    all(index -> index in axes(matrix, 1), selected) || throw(ArgumentError(
        "requested component is outside the fitted component range"))
    images = cat((reshape(matrix[index, :], shape) for index in selected)...; dims=3)
    rows = cld(length(selected), columns)
    montage, centers = _tile_montage(images, rows, Int(columns))
    centers = [(x, y, "C$(selected[index])") for (index, (x, y, _)) in enumerate(centers)]
    extreme = maximum(abs, filter(isfinite, montage))
    nonnegative = all(>=(0), matrix[selected, :])
    colorrange = nonnegative ? (zero(eltype(matrix)), maximum(matrix[selected, :])) :
                 (-extreme, extreme)
    return componentplot(ComponentData(montage, centers, colorrange, String(title));
                         colormap=nonnegative ? :magma : :balance, kwargs...)
end

function reconstructionplot(fitted, X::AbstractMatrix; shape::Tuple{Int,Int},
                            observations=1:min(6, size(X, 1)), kwargs...)
    selected = collect(observations)
    prod(shape) == size(X, 2) || throw(DimensionMismatch(
        "reconstruction shape does not match the feature count"))
    original = Matrix(X[selected, :])
    reconstructed = Tilia.inverse_transform(fitted, Tilia.transform(fitted, original))
    error = abs.(original .- reconstructed)
    count = length(selected)
    height, width = shape
    montage = fill(convert(float(eltype(X)), NaN),
                   3 * height + 2, count * width + count - 1)
    labels = Tuple{Float64,Float64,String}[]
    for (column, observation) in enumerate(selected)
        x = (column - 1) * (width + 1) + 1
        montage[1:height, x:x + width - 1] .= reshape(original[column, :], shape)
        montage[height + 2:2height + 1, x:x + width - 1] .=
            reshape(reconstructed[column, :], shape)
        montage[2height + 3:3height + 2, x:x + width - 1] .= reshape(error[column, :], shape)
        push!(labels, (x + (width - 1) / 2, 1.0, "#$observation"))
    end
    finite = filter(isfinite, montage)
    return reconstructionplot(ReconstructionData(montage, labels,
                              (minimum(finite), maximum(finite))); kwargs...)
end

Makie.@recipe ProjectionPlot (data,) begin
    color = :steelblue
    markersize = 10
    alpha = 0.75
    show_centroids = true
end

Makie.@recipe ScreePlot (data,) begin
    barcolor = :cornflowerblue
    linecolor = :darkorange
    thresholdcolor = :gray45
end

Makie.@recipe Biplot (data,) begin
    color = :steelblue
    markersize = 9
    alpha = 0.65
    loadingcolor = :crimson
end

Makie.@recipe LoadingsPlot (data,) begin
    positivecolor = :seagreen
    negativecolor = :indianred
end

Makie.@recipe ComponentPlot (data,) begin
    colormap = :balance
end

Makie.@recipe ReconstructionPlot (data,) begin
    colormap = :viridis
end

Makie.plottype(::ProjectionData) = ProjectionPlot
Makie.plottype(::ScreeData) = ScreePlot
Makie.plottype(::BiplotData) = Biplot
Makie.plottype(::LoadingsData) = LoadingsPlot
Makie.plottype(::ComponentData) = ComponentPlot
Makie.plottype(::ReconstructionData) = ReconstructionPlot

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, data::ProjectionData)
    return (title=data.title,
            xlabel=_component_label(data.components[1], data.variance_ratio),
            ylabel=_component_label(data.components[2], data.variance_ratio))
end

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::ScreeData) =
    (title="Explained variance", xlabel="Component", ylabel="Variance fraction",
     limits=(nothing, nothing, 0, 1.02))

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, data::BiplotData)
    return (title="PCA biplot",
            xlabel=_component_label(data.components[1], data.variance_ratio),
            ylabel=_component_label(data.components[2], data.variance_ratio))
end

function Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, data::LoadingsData)
    ticks = (collect(eachindex(data.names)), data.names)
    return (title="Component $(data.component) loadings", xlabel="Feature",
            ylabel="Loading", xticks=ticks, xticklabelrotation=π / 4)
end

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, data::ComponentData) =
    (title=data.title, aspect=Makie.DataAspect(), yreversed=true,
     xgridvisible=false, ygridvisible=false, xticksvisible=false, yticksvisible=false,
     xticklabelsvisible=false, yticklabelsvisible=false,
     bottomspinevisible=false, topspinevisible=false,
     leftspinevisible=false, rightspinevisible=false)

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::ReconstructionData) =
    (title="Original · reconstruction · absolute error", aspect=Makie.DataAspect(),
     yreversed=true, xgridvisible=false, ygridvisible=false,
     xticksvisible=false, yticksvisible=false,
     xticklabelsvisible=false, yticklabelsvisible=false,
     bottomspinevisible=false, topspinevisible=false,
     leftspinevisible=false, rightspinevisible=false)

function Makie.plot!(plot::ProjectionPlot)
    data = plot.data[]
    colors, group_info = _group_colors(data.groups)
    Makie.scatter!(plot, data.scores[:, 1], data.scores[:, 2]; color=colors,
                   markersize=plot.markersize, alpha=plot.alpha)
    if plot.show_centroids[] && group_info !== nothing
        levels, lookup = group_info
        for level in levels
            indices = findall(==(level), data.groups)
            center = vec(sum(data.scores[indices, :]; dims=1) / length(indices))
            Makie.scatter!(plot, [center[1]], [center[2]]; color=lookup[level],
                           marker=:star5, markersize=22, strokecolor=:white,
                           strokewidth=1.5)
            Makie.text!(plot, center[1], center[2]; text="  $(level)",
                        color=lookup[level], fontsize=13, font=:bold)
        end
    end
    return plot
end

function Makie.plot!(plot::ScreePlot)
    data = plot.data[]
    positions = collect(eachindex(data.explained))
    Makie.barplot!(plot, positions, data.explained; color=plot.barcolor)
    Makie.scatterlines!(plot, positions, data.cumulative; color=plot.linecolor,
                        marker=:circle, markersize=8, linewidth=2.5)
    Makie.hlines!(plot, [data.threshold]; color=plot.thresholdcolor, linestyle=:dash)
    reached = findfirst(>=(data.threshold), data.cumulative)
    reached === nothing || Makie.vlines!(plot, [reached]; color=plot.thresholdcolor,
                                         linestyle=:dot)
    return plot
end


function Makie.plot!(plot::Biplot)
    data = plot.data[]
    colors, _ = _group_colors(data.groups)
    Makie.scatter!(plot, data.scores[:, 1], data.scores[:, 2]; color=colors,
                   markersize=plot.markersize, alpha=plot.alpha)
    score_scale = 0.75 * min(maximum(abs, data.scores[:, 1]),
                             maximum(abs, data.scores[:, 2]))
    loading_scale = maximum(abs, data.loadings)
    arrows = data.loadings .* (score_scale / max(loading_scale, eps()))
    Makie.arrows2d!(plot, zeros(size(arrows, 1)), zeros(size(arrows, 1)),
                    arrows[:, 1], arrows[:, 2]; color=plot.loadingcolor,
                    shaftwidth=1.5, tipwidth=8, tiplength=10)
    Makie.text!(plot, arrows[:, 1], arrows[:, 2]; text=data.feature_names,
                color=plot.loadingcolor, fontsize=12, align=(:center, :bottom))
    return plot
end

function Makie.plot!(plot::LoadingsPlot)
    data = plot.data[]
    colors = ifelse.(data.values .>= 0, plot.positivecolor[], plot.negativecolor[])
    Makie.barplot!(plot, collect(eachindex(data.values)), data.values; color=colors)
    Makie.hlines!(plot, [0]; color=:gray45, linewidth=1)
    return plot
end

function Makie.plot!(plot::ComponentPlot)
    data = plot.data[]
    Makie.heatmap!(plot, data.montage; colormap=plot.colormap,
                   colorrange=data.colorrange, nan_color=:white)
    for (x, y, label) in data.centers
        Makie.text!(plot, x, y; text=label, align=(:center, :bottom),
                    color=:black, fontsize=12, font=:bold)
    end
    return plot
end

function Makie.plot!(plot::ReconstructionPlot)
    data = plot.data[]
    Makie.heatmap!(plot, data.montage; colormap=plot.colormap,
                   colorrange=data.colorrange, nan_color=:white)
    height = (size(data.montage, 1) - 2) ÷ 3
    for (x, label) in ((height / 2, "Original"), (1.5height + 1, "Reconstructed"),
                       (2.5height + 2, "Absolute error"))
        Makie.text!(plot, x, 0; text=label, align=(:center, :bottom),
                    color=:black, fontsize=12, font=:bold)
    end
    return plot
end
