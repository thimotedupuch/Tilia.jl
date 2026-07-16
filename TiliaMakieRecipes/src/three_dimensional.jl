_matrix3(X) = size(X, 2) == 3 ? Matrix(X) :
    throw(DimensionMismatch("this plot requires exactly three coordinates"))

function _axis3(fig, position; title, labels)
    Makie.Axis3(fig[position...]; title=title, xlabel=labels[1], ylabel=labels[2],
                zlabel=labels[3], aspect=:data, perspectiveness=0.55,
                azimuth=1.25pi, elevation=0.22pi)
end

function _scatter_groups3!(ax, points, groups; markersize=9)
    if groups === nothing
        Makie.scatter!(ax, points[:,1], points[:,2], points[:,3];
                       color=:steelblue, markersize=markersize)
        return
    end
    for (i, level) in enumerate(_levels(groups))
        mask = groups .== level
        Makie.scatter!(ax, points[mask,1], points[mask,2], points[mask,3];
                       color=_palette(i), markersize=markersize, label=string(level))
    end
    Makie.axislegend(ax; position=:rt)
end

"""Three-component dimensionality-reduction projection on a Makie `Axis3`."""
function projectionplot3d(fitted, X; groups=nothing, components=(1,2,3),
                          title=string(nameof(typeof(fitted.model))), markersize=9)
    length(components) == 3 || throw(ArgumentError("components must contain three indices"))
    transformed = Tilia.transform(fitted, X)
    maximum(components) <= size(transformed,2) || throw(ArgumentError(
        "requested projection component exceeds the fitted component count"))
    points = Matrix(transformed[:,collect(components)])
    groups === nothing || length(groups)==size(points,1) || throw(DimensionMismatch(
        "projection groups must match the observation count"))
    ratios = hasproperty(fitted,:explained_variance_ratio) ? fitted.explained_variance_ratio : nothing
    labels = Tuple(_component_label(component, ratios) for component in components)
    fig=Makie.Figure(size=(850,700)); ax=_axis3(fig,(1,1); title=title, labels=labels)
    _scatter_groups3!(ax,points,groups;markersize=markersize); fig
end

"""General labeled point-cloud viewer, optionally with indexed graph edges."""
function pointcloudplot(points; groups=nothing, values=nothing, edges=nothing,
                        labels=("X","Y","Z"), title="Point cloud", markersize=9)
    points=_matrix3(points); n=size(points,1)
    groups === nothing || length(groups)==n || throw(DimensionMismatch("groups must match points"))
    values === nothing || length(values)==n || throw(DimensionMismatch("values must match points"))
    groups === nothing || values === nothing || throw(ArgumentError("choose groups or scalar values, not both"))
    fig=Makie.Figure(size=(850,700)); ax=_axis3(fig,(1,1);title=title,labels=labels)
    if edges !== nothing
        for edge in edges
            i,j=edge
            Makie.lines!(ax,points[[i,j],1],points[[i,j],2],points[[i,j],3];color=(:gray45,0.28),linewidth=1)
        end
    end
    if values !== nothing
        plot=Makie.scatter!(ax,points[:,1],points[:,2],points[:,3];color=values,colormap=:viridis,markersize=markersize)
        Makie.Colorbar(fig[1,2],plot,label="Value")
    else
        _scatter_groups3!(ax,points,groups;markersize=markersize)
    end
    fig
end

function _empirical_covariance3(points, center)
    centered=points .- permutedims(center)
    transpose(centered)*centered/max(size(points,1)-1,1)
end

function _ellipsoid!(ax, center, covariance; color=:steelblue, scale=2.0)
    decomposition=eigen(Symmetric(Matrix(covariance)))
    radii=scale .* sqrt.(max.(decomposition.values,0))
    u=range(0,2pi;length=48); v=range(0,pi;length=25)
    x=Matrix{Float64}(undef,length(u),length(v)); y=similar(x); z=similar(x)
    for i in eachindex(u), j in eachindex(v)
        localpoint=radii .* [cos(u[i])*sin(v[j]),sin(u[i])*sin(v[j]),cos(v[j])]
        point=center + decomposition.vectors*localpoint
        x[i,j],y[i,j],z[i,j]=point
    end
    Makie.surface!(ax,x,y,z;color=(color,0.13),transparency=true,shading=Makie.NoShading)
end

"""Three-dimensional clustering with centers, noise, and covariance ellipsoids."""
function clusterplot3d(fitted, X; covariance_ellipsoids=true,
                       title="Three-dimensional cluster structure")
    X=_matrix3(X); labels=_labels_for(fitted,X); levels=_levels(labels)
    fig=Makie.Figure(size=(880,720)); ax=_axis3(fig,(1,1);title=title,labels=("Feature 1","Feature 2","Feature 3"))
    for (i,level) in enumerate(levels)
        mask=labels.==level
        if level==0
            Makie.scatter!(ax,X[mask,1],X[mask,2],X[mask,3];marker=:xcross,color=:gray20,markersize=14,label="Noise")
        else
            Makie.scatter!(ax,X[mask,1],X[mask,2],X[mask,3];color=_palette(i),markersize=8,label="Cluster $(level)")
        end
    end
    centers = hasproperty(fitted,:centers) ? fitted.centers : hasproperty(fitted,:means) ? fitted.means :
        reduce(vcat,[permutedims([_mean(X[labels.==level,j]) for j in 1:3]) for level in filter(!=(0),levels)])
    Makie.scatter!(ax,centers[:,1],centers[:,2],centers[:,3];marker=:star5,color=:white,strokecolor=:black,strokewidth=2,markersize=24,label="Center")
    if covariance_ellipsoids
        for (i,level) in enumerate(filter(!=(0),levels))
            covariance=hasproperty(fitted,:covariances) ? fitted.covariances[i][1:3,1:3] : _empirical_covariance3(X[labels.==level,:],centers[i,:])
            _ellipsoid!(ax,centers[i,:],covariance;color=_palette(i))
        end
    end
    Makie.axislegend(ax;position=:rt,merge=true); fig
end

"""Three-dimensional nearest-neighbor graph between queries and training points."""
function neighborhoodplot3d(fitted, queries; n_neighbors=5,title="Nearest-neighbor neighborhoods")
    train=_matrix3(fitted.training_data); queries=_matrix3(queries)
    _,indices=Tilia.kneighbors(fitted,queries;n_neighbors=n_neighbors)
    fig=Makie.Figure(size=(850,700)); ax=_axis3(fig,(1,1);title=title,labels=("Feature 1","Feature 2","Feature 3"))
    Makie.scatter!(ax,train[:,1],train[:,2],train[:,3];color=(:gray45,0.55),markersize=7,label="Training")
    for q in axes(queries,1), j in axes(indices,2)
        i=indices[q,j]
        Makie.lines!(ax,[queries[q,1],train[i,1]],[queries[q,2],train[i,2]],[queries[q,3],train[i,3]];color=(:steelblue,0.4),linewidth=1.5)
    end
    Makie.scatter!(ax,queries[:,1],queries[:,2],queries[:,3];marker=:star5,color=:darkorange,markersize=24,label="Query")
    Makie.axislegend(ax); fig
end

"""Regression response surface for a fitted model with exactly two predictors."""
function regressionsurfaceplot(fitted,X,y;resolution=70,title="Regression response surface")
    X=_matrix2(X); actual=vec(y); length(actual)==size(X,1) || throw(DimensionMismatch("target must match observations"))
    xs,ys,grid=_grid2(X;resolution=resolution); predicted=Tilia.predict(fitted,grid)
    z=reshape(predicted,length(xs),length(ys))
    fig=Makie.Figure(size=(880,720)); ax=_axis3(fig,(1,1);title=title,labels=("Feature 1","Feature 2","Response"))
    Makie.surface!(ax,xs,ys,z;color=z,colormap=:viridis,alpha=0.72,transparency=true)
    fitted_values=vec(Tilia.predict(fitted,X)); residuals=actual.-fitted_values
    Makie.scatter!(ax,X[:,1],X[:,2],actual;color=residuals,colormap=:coolwarm,markersize=9)
    for i in axes(X,1); Makie.lines!(ax,[X[i,1],X[i,1]],[X[i,2],X[i,2]],[fitted_values[i],actual[i]];color=(:gray20,0.35)); end
    fig
end

"""Three-dimensional score landscape for two numeric tuning parameters."""
function tuninglandscapeplot(result; xparameter,yparameter,title="Tuning-score landscape")
    xs=sort(unique(Float64(_trialparam(t,xparameter)) for t in result.trials)); ys=sort(unique(Float64(_trialparam(t,yparameter)) for t in result.trials)); z=fill(NaN,length(xs),length(ys))
    for t in result.trials; x=Float64(_trialparam(t,xparameter)); y=Float64(_trialparam(t,yparameter)); z[findfirst(==(x),xs),findfirst(==(y),ys)]=t.score; end
    fig=Makie.Figure(size=(880,720)); ax=_axis3(fig,(1,1);title=title,labels=(string(xparameter),string(yparameter),"Score"))
    Makie.surface!(ax,xs,ys,z;color=z,colormap=:viridis,alpha=0.82,transparency=true)
    for t in result.trials; Makie.scatter!(ax,[Float64(_trialparam(t,xparameter))],[Float64(_trialparam(t,yparameter))],[t.score];color=:white,strokecolor=:black,strokewidth=1,markersize=10); end
    fig
end
