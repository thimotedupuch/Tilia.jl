function _prediction_response(fitted, data, target)
    if hasproperty(fitted, :classes) && fitted.classes !== nothing
        classes = collect(fitted.classes)
        isempty(classes) && return vec(Tilia.predict(fitted, data)), nothing
        selected = target === nothing ? last(classes) : target
        index = findfirst(==(selected), classes)
        index === nothing && throw(ArgumentError("target class $(repr(selected)) is not present in the fitted model"))
        return vec(Tilia.predict_proba(fitted, data)[:, index]), string(selected)
    end
    target === nothing || throw(ArgumentError("target is only valid for probabilistic classifiers"))
    vec(Tilia.predict(fitted, data)), nothing
end

function _feature_grid(column, resolution)
    lo, hi = extrema(column)
    lo == hi && return fill(float(lo), resolution)
    collect(range(float(lo), float(hi); length=resolution))
end

"""
Plot partial dependence for one or two features.

One-feature plots optionally include individual conditional expectation (ICE)
curves. Two-feature plots use a filled contour by default; pass `surface=true`
for an interactive `Axis3` surface.
"""
function partialdependenceplot(fitted, X; feature, target=nothing, resolution=40,
                               ice=true, observations=min(30,size(X,1)),
                               feature_names=nothing, surface=false,
                               title="Partial dependence")
    resolution >= 2 || throw(ArgumentError("resolution must be at least two"))
    names = feature_names === nothing ? ["Feature $(i)" for i in axes(X,2)] : string.(feature_names)
    length(names) == size(X,2) || throw(DimensionMismatch("feature_names must match the feature count"))
    features = feature isa Integer ? (Int(feature),) : Tuple(Int.(feature))
    all(i -> i in axes(X,2), features) || throw(ArgumentError("feature index is outside the input columns"))
    length(features) in (1,2) || throw(ArgumentError("feature must identify one or two columns"))
    data = Matrix(X)

    if length(features) == 1
        index = only(features); grid = _feature_grid(view(data,:,index),resolution)
        chosen = observations isa Integer ? collect(round.(Int,range(1,size(data,1);length=min(observations,size(data,1))))) : collect(observations)
        all(i -> i in axes(data,1), chosen) || throw(ArgumentError("ICE observation index is outside the input rows"))
        ice_values = Matrix{Float64}(undef,length(chosen),length(grid))
        dependence = Vector{Float64}(undef,length(grid)); response_label = nothing
        for (j,value) in enumerate(grid)
            varied=copy(data); varied[:,index].=value
            response,response_label=_prediction_response(fitted,varied,target)
            dependence[j]=_mean(response); ice_values[:,j].=response[chosen]
        end
        fig=Makie.Figure(size=(780,540)); ylabel=response_label===nothing ? "Partial dependence" : "P($(response_label))"
        ax=Makie.Axis(fig[1,1],title=title,xlabel=names[index],ylabel=ylabel)
        if ice
            for row in axes(ice_values,1); Makie.lines!(ax,grid,ice_values[row,:];color=(:gray45,0.18),linewidth=1); end
        end
        Makie.lines!(ax,grid,dependence;color=:crimson,linewidth=4,label="Partial dependence")
        baseline = min(minimum(dependence), minimum(ice_values))
        Makie.scatter!(ax,data[:,index],fill(baseline,size(data,1));
                       color=(:black,0.24),marker=:vline,markersize=9)
        Makie.axislegend(ax;position=:lt); return fig
    end

    first_feature, second_feature = features
    xs=_feature_grid(view(data,:,first_feature),resolution); ys=_feature_grid(view(data,:,second_feature),resolution)
    dependence=Matrix{Float64}(undef,length(xs),length(ys)); response_label=nothing
    for i in eachindex(xs), j in eachindex(ys)
        varied=copy(data); varied[:,first_feature].=xs[i]; varied[:,second_feature].=ys[j]
        response,response_label=_prediction_response(fitted,varied,target); dependence[i,j]=_mean(response)
    end
    zlabel=response_label===nothing ? "Partial dependence" : "P($(response_label))"
    fig=Makie.Figure(size=(820,650))
    if surface
        ax=_axis3(fig,(1,1);title=title,labels=(names[first_feature],names[second_feature],zlabel))
        Makie.surface!(ax,xs,ys,dependence;color=dependence,colormap=:viridis)
    else
        ax=Makie.Axis(fig[1,1],title=title,xlabel=names[first_feature],ylabel=names[second_feature])
        contour=Makie.contourf!(ax,xs,ys,dependence;levels=18,colormap=:viridis)
        Makie.scatter!(ax,data[:,first_feature],data[:,second_feature];color=(:white,0.28),markersize=5)
        Makie.Colorbar(fig[1,2],contour;label=zlabel)
    end
    fig
end

function _simplex_grid!(ax)
    height=sqrt(3)/2
    Makie.lines!(ax,[0,1,0.5,0],[0,0,height,0];color=:gray20,linewidth=2)
    for fraction in 0.2:0.2:0.8
        Makie.lines!(ax,[fraction,0.5+0.5fraction],[0,height*(1-fraction)];color=(:gray45,0.18))
        Makie.lines!(ax,[1-fraction,0.5-0.5fraction],[0,height*(1-fraction)];color=(:gray45,0.18))
        Makie.lines!(ax,[0.5fraction,1-0.5fraction],[height*fraction,height*fraction];color=(:gray45,0.18))
    end
end

"""Ternary plot of predicted probabilities from an exactly three-class classifier."""
function probabilitysimplexplot(probabilities::AbstractMatrix; classes=nothing, groups=nothing,
                                title="Classifier probability simplex", markersize=10)
    size(probabilities,2)==3 || throw(DimensionMismatch("probability simplex requires exactly three columns"))
    all(isfinite,probabilities) && all(p -> p >= 0,probabilities) || throw(ArgumentError("probabilities must be finite and nonnegative"))
    rowsums=vec(sum(probabilities;dims=2)); all(>(0),rowsums) || throw(ArgumentError("each probability row must have positive mass"))
    normalized=probabilities ./ rowsums
    classes=classes===nothing ? ["Class 1","Class 2","Class 3"] : string.(classes)
    length(classes)==3 || throw(DimensionMismatch("classes must contain three labels"))
    groups===nothing || length(groups)==size(normalized,1) || throw(DimensionMismatch("groups must match observations"))
    height=sqrt(3)/2; x=normalized[:,2].+0.5normalized[:,3]; y=height.*normalized[:,3]
    fig=Makie.Figure(size=(760,680),backgroundcolor=:white)
    ax=Makie.Axis(fig[1,1],title=title,aspect=Makie.DataAspect(),backgroundcolor=:white)
    Makie.hidedecorations!(ax); Makie.hidespines!(ax); _simplex_grid!(ax)
    plotgroups = groups===nothing ? [argmax(view(normalized,i,:)) for i in axes(normalized,1)] : groups
    for (i,level) in enumerate(_levels(plotgroups))
        mask=plotgroups.==level
        Makie.scatter!(ax,x[mask],y[mask];color=_palette(i),markersize=markersize,strokecolor=:white,strokewidth=0.6,label=string(level))
    end
    Makie.text!(ax,0,-0.035;text=classes[1],align=(:center,:top),font=:bold)
    Makie.text!(ax,1,-0.035;text=classes[2],align=(:center,:top),font=:bold)
    Makie.text!(ax,0.5,height+0.035;text=classes[3],align=(:center,:bottom),font=:bold)
    Makie.xlims!(ax,-0.08,1.08); Makie.ylims!(ax,-0.09,height+0.10); Makie.axislegend(ax;position=:rt); fig
end

function probabilitysimplexplot(fitted, X; groups=nothing, classes=nothing, kwargs...)
    probabilities=Tilia.predict_proba(fitted,X)
    labels=classes===nothing && hasproperty(fitted,:classes) ? fitted.classes : classes
    probabilitysimplexplot(probabilities;classes=labels,groups=groups,kwargs...)
end
