const _DIAGNOSTIC_COLORS = Makie.wong_colors()

_matrix2(X) = size(X, 2) == 2 ? Matrix(X) : throw(DimensionMismatch("this plot requires exactly two features"))
_palette(i) = _DIAGNOSTIC_COLORS[mod1(i, length(_DIAGNOSTIC_COLORS))]
function _levels(y)
    values = unique(y)
    try
        sort!(values)
    catch
        # Some user-defined class labels are equality-comparable but not orderable.
    end
    values
end
_encode(y, levels=_levels(y)) = [findfirst(==(v), levels) for v in y]
_padlimits(v) = begin d = maximum(v) - minimum(v); p = iszero(d) ? 1.0 : 0.08d; (minimum(v)-p, maximum(v)+p) end

function _grid2(X; resolution=160)
    xlim, ylim = _padlimits(view(X, :, 1)), _padlimits(view(X, :, 2))
    xs, ys = range(xlim...; length=resolution), range(ylim...; length=resolution)
    grid = [Float64[x, y] for y in ys for x in xs]
    xs, ys, reduce(vcat, permutedims.(grid))
end

function _labels_for(fitted, X)
    if hasproperty(fitted, :labels) && length(fitted.labels) == size(X, 1)
        return fitted.labels
    end
    return Tilia.predict(fitted, X)
end

function _centers_for(fitted, X, labels)
    hasproperty(fitted, :centers) && return fitted.centers
    hasproperty(fitted, :means) && return fitted.means
    levels = filter(!=(0), _levels(labels))
    reduce(vcat, [permutedims([sum(X[labels .== k, j]) / count(==(k), labels) for j in 1:2]) for k in levels])
end

function _ellipse!(ax, center, covariance; color=:black, scale=2.0)
    a, b, d = covariance[1,1], covariance[1,2], covariance[2,2]
    delta = sqrt((a-d)^2 + 4b^2)
    l1, l2 = max((a+d+delta)/2, 0), max((a+d-delta)/2, 0)
    angle = 0.5atan(2b, a-d)
    t = range(0, 2pi; length=121)
    ca, sa = cos(angle), sin(angle)
    x = center[1] .+ scale .* (sqrt(l1).*cos.(t).*ca .- sqrt(l2).*sin.(t).*sa)
    y = center[2] .+ scale .* (sqrt(l1).*cos.(t).*sa .+ sqrt(l2).*sin.(t).*ca)
    Makie.lines!(ax, x, y; color=color, linewidth=2)
end

"""Plot two-dimensional cluster assignments, regions, centers, ellipses, and noise."""
function clusterplot(fitted, X; decision_regions=true, covariance_ellipses=true,
                     resolution=160, title="Cluster structure")
    X = _matrix2(X); labels = _labels_for(fitted, X); levels = _levels(labels)
    fig = Makie.Figure(size=(760, 620)); ax = Makie.Axis(fig[1,1], title=title, xlabel="Feature 1", ylabel="Feature 2")
    if decision_regions && !(0 in levels)
        xs, ys, grid = _grid2(X; resolution)
        z = reshape(_encode(Tilia.predict(fitted, grid), levels), length(xs), length(ys))
        Makie.contourf!(ax, xs, ys, z; levels=length(levels), colormap=[:transparent; [(_palette(i), 0.16) for i in 1:length(levels)]])
    end
    for (i, level) in enumerate(levels)
        mask = labels .== level
        level == 0 ? Makie.scatter!(ax, X[mask,1], X[mask,2]; color=:gray25, marker=:xcross, markersize=13, label="Noise") :
            Makie.scatter!(ax, X[mask,1], X[mask,2]; color=_palette(i), markersize=10, label="Cluster $(level)")
    end
    centers = _centers_for(fitted, X, labels)
    Makie.scatter!(ax, centers[:,1], centers[:,2]; marker=:star5, color=:white, strokecolor=:black, strokewidth=2, markersize=24, label="Center")
    if covariance_ellipses
        nonnoise = filter(!=(0), levels)
        for (i, level) in enumerate(nonnoise)
            cov = if hasproperty(fitted, :covariances)
                fitted.covariances[i][1:2,1:2]
            else
                pts = X[labels .== level, :]; c = centers[i, :]; n = max(size(pts,1)-1, 1)
                [(sum((pts[:,r].-c[r]).*(pts[:,s].-c[s]))/n) for r in 1:2, s in 1:2]
            end
            _ellipse!(ax, centers[i,:], cov; color=_palette(i))
        end
    end
    Makie.axislegend(ax; position=:rt, merge=true); fig
end

"""Render the merge tree stored by agglomerative clustering or feature agglomeration."""
function dendrogram(fitted; labels=nothing, title="Dendrogram")
    children, heights = fitted.children, fitted.merge_distances
    n = hasproperty(fitted, :labels) ? length(fitted.labels) : size(children,1)+1
    labels === nothing && (labels = string.(1:n))
    xpos = Dict(i => Float64(i) for i in 1:n); ypos = Dict(i => 0.0 for i in 1:n)
    fig = Makie.Figure(size=(max(700, 24n), 520)); ax = Makie.Axis(fig[1,1], title=title, ylabel="Merge distance", xticks=(1:n, string.(labels)))
    for i in axes(children,1)
        left, right = children[i,1], children[i,2]; h = heights[i]
        xl, xr = xpos[left], xpos[right]; yl, yr = ypos[left], ypos[right]
        Makie.lines!(ax, [xl,xl,xr,xr], [yl,h,h,yr]; color=:steelblue, linewidth=2)
        xpos[n+i] = (xl+xr)/2; ypos[n+i] = h
    end
    fig
end

"""Plot the prediction regions of any fitted classifier accepting two columns."""
function decisionboundaryplot(fitted, X, y; resolution=180, title="Decision boundary")
    X = _matrix2(X); levels = _levels(y); xs, ys, grid = _grid2(X; resolution)
    z = reshape(_encode(Tilia.predict(fitted, grid), levels), length(xs), length(ys))
    fig = Makie.Figure(size=(720,600)); ax = Makie.Axis(fig[1,1], title=title, xlabel="Feature 1", ylabel="Feature 2")
    Makie.contourf!(ax, xs, ys, z; levels=length(levels), colormap=[(_palette(i),0.22) for i in eachindex(levels)])
    for (i,l) in enumerate(levels); m=y.==l; Makie.scatter!(ax,X[m,1],X[m,2];color=_palette(i),strokecolor=:white,strokewidth=0.7,label=string(l)); end
    Makie.axislegend(ax); fig
end

function _tree_positions(nodes)
    x=Dict{Int,Float64}(); y=Dict{Int,Float64}(); leaf=Ref(0)
    function visit(i,d)
        node=nodes[i]; y[i]=-d
        if node.is_leaf; leaf[]+=1; x[i]=leaf[]
        else; visit(node.left,d+1); visit(node.right,d+1); x[i]=(x[node.left]+x[node.right])/2; end
    end
    visit(1,0); x,y
end

"""Visualize a fitted decision tree with split, sample, impurity, and prediction labels."""
function treeplot(fitted; feature_names=nothing, title="Decision tree")
    nodes=fitted.nodes; feature_names===nothing && (feature_names=["x$(i)" for i in 1:length(fitted.feature_importances)])
    x,y=_tree_positions(nodes); fig=Makie.Figure(size=(1200,700)); ax=Makie.Axis(fig[1,1],title=title); Makie.hidedecorations!(ax); Makie.hidespines!(ax)
    for (i,n) in enumerate(nodes)
        if !n.is_leaf
            for child in (n.left,n.right); Makie.lines!(ax,[x[i],x[child]],[y[i],y[child]];color=:gray55,linewidth=2); end
        end
    end
    for (i,n) in enumerate(nodes)
        pred = n.predicted_class > 0 && hasproperty(fitted,:classes) ? fitted.classes[n.predicted_class] : round(n.prediction,digits=3)
        head = n.is_leaf ? "predict = $(pred)" : "$(feature_names[n.feature]) ≤ $(round(n.threshold,digits=3))"
        text = "$(head)\nn = $(n.samples)  impurity = $(round(n.impurity,digits=3))"
        color = n.predicted_class > 0 ? (_palette(n.predicted_class),0.88) : (:steelblue,0.82)
        Makie.scatter!(ax,[x[i]],[y[i]];marker=:rect,markersize=(220,58),color=color,strokecolor=:white,strokewidth=2)
        Makie.text!(ax,x[i],y[i];text=text,align=(:center,:center),fontsize=11,color=:white)
    end
    Makie.xlims!(ax, 0, maximum(values(x)) + 1)
    Makie.ylims!(ax, minimum(values(y)) - 0.35, 0.35)
    fig
end

_score(y, p) = eltype(y) <: Number && length(unique(y)) > max(20, length(y)÷5) ? -sqrt(sum(abs2, y.-p)/length(y)) : sum(y.==p)/length(y)

function _learning_order(y)
    levels = unique(y)
    length(levels) > max(20, length(y) ÷ 5) && return collect(eachindex(y))
    buckets = [findall(==(level), y) for level in levels]
    [buckets[k][i] for i in 1:maximum(length.(buckets)) for k in eachindex(buckets) if i <= length(buckets[k])]
end

"""Plot training and held-out score as progressively more observations are used."""
function learningcurveplot(model, X, y; train_sizes=range(0.15,1;length=8), validation_fraction=0.25, title="Learning curve")
    order=_learning_order(y); X=X[order,:]; y=y[order]
    n=size(X,1); cut=max(2,floor(Int,n*(1-validation_fraction))); sizes=unique(max.(2,round.(Int,collect(train_sizes).*cut)))
    tr=Float64[]; va=Float64[]
    for s in sizes; fitted=Tilia.fit(model,X[1:s,:],y[1:s]); push!(tr,_score(y[1:s],Tilia.predict(fitted,X[1:s,:]))); push!(va,_score(y[cut+1:end],Tilia.predict(fitted,X[cut+1:end,:]))); end
    fig=Makie.Figure(size=(720,500)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Training observations",ylabel="Score")
    Makie.scatterlines!(ax,sizes,tr;label="Training",color=:steelblue); Makie.scatterlines!(ax,sizes,va;label="Validation",color=:darkorange); Makie.axislegend(ax); fig
end

"""Plot train and validation scores while varying one estimator parameter."""
function validationcurveplot(model, X, y; parameter, values, validation_fraction=0.25, title="Validation curve")
    order=_learning_order(y); X=X[order,:]; y=y[order]
    n=size(X,1); cut=max(2,floor(Int,(1-validation_fraction)*n)); tr=Float64[]; va=Float64[]
    for value in values
        changed=Tilia._replace_parameters(model, NamedTuple{(Symbol(parameter),)}((value,)))
        fitted=Tilia.fit(changed,X[1:cut,:],y[1:cut]); push!(tr,_score(y[1:cut],Tilia.predict(fitted,X[1:cut,:]))); push!(va,_score(y[cut+1:end],Tilia.predict(fitted,X[cut+1:end,:])))
    end
    fig=Makie.Figure(size=(720,500)); ax=Makie.Axis(fig[1,1],title=title,xlabel=string(parameter),ylabel="Score",xticks=(1:length(values),string.(values)))
    Makie.scatterlines!(ax,1:length(values),tr;label="Training",color=:steelblue); Makie.scatterlines!(ax,1:length(values),va;label="Validation",color=:darkorange); Makie.axislegend(ax); fig
end

function _regression_parts(fitted,X,y)
    predicted=vec(Tilia.predict(fitted,X)); actual=vec(y); residuals=actual.-predicted; actual,predicted,residuals
end

"""Three-panel regression diagnostic: actual vs predicted, residuals, and distribution."""
function residualplot(fitted,X,y; title="Regression diagnostics")
    actual,predicted,residuals=_regression_parts(fitted,X,y); fig=Makie.Figure(size=(1120,370)); Makie.Label(fig[0,1:3],title,fontsize=22)
    a=Makie.Axis(fig[1,1],title="Predicted versus actual",xlabel="Actual",ylabel="Predicted"); Makie.scatter!(a,actual,predicted;color=residuals,colormap=:coolwarm); lo=min(minimum(actual),minimum(predicted)); hi=max(maximum(actual),maximum(predicted)); Makie.lines!(a,[lo,hi],[lo,hi];color=:gray40,linestyle=:dash)
    b=Makie.Axis(fig[1,2],title="Residual plot",xlabel="Predicted",ylabel="Residual"); Makie.scatter!(b,predicted,residuals;color=:steelblue); Makie.hlines!(b,[0];color=:gray40,linestyle=:dash)
    c=Makie.Axis(fig[1,3],title="Residual distribution",xlabel="Residual",ylabel="Count"); Makie.hist!(c,residuals;bins=20,color=:slateblue); Makie.vlines!(c,[0];color=:gray40,linestyle=:dash); fig
end

predictedactualplot(fitted,X,y;kwargs...) = residualplot(fitted,X,y;kwargs...)
residualdistributionplot(fitted,X,y;kwargs...) = residualplot(fitted,X,y;kwargs...)

function _coefficients(fitted)
    hasproperty(fitted,:coefficients) || throw(ArgumentError("fitted model does not expose coefficients")); fitted.coefficients
end

"""Compare signed fitted coefficients across features and, for classifiers, classes."""
function coefficientplot(fitted; feature_names=nothing,title="Model coefficients")
    c=_coefficients(fitted); C=ndims(c)==1 ? reshape(c,1,:) : (size(c,2)==length(c) ? reshape(c,1,:) : c); p=size(C,2)
    feature_names===nothing && (feature_names=["x$(i)" for i in 1:p]); fig=Makie.Figure(size=(max(720,55p),500)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Feature",ylabel="Coefficient",xticks=(1:p,string.(feature_names)),xticklabelrotation=pi/4)
    for r in axes(C,1); Makie.scatterlines!(ax,1:p,vec(C[r,:]);color=_palette(r),label=size(C,1)>1 ? "Class $(r)" : "Coefficient"); end
    Makie.hlines!(ax,[0];color=:gray50,linestyle=:dash); size(C,1)>1 && Makie.axislegend(ax); fig
end

"""Trace Lasso or elastic-net coefficients across regularization strengths."""
function regularizationpathplot(model,X,y; lambdas=10.0.^range(-3,1;length=30),feature_names=nothing,title="Regularization path")
    paths=Matrix{Float64}(undef,length(lambdas),size(X,2))
    for (i,l) in enumerate(lambdas); changed=Tilia._replace_parameters(model,(lambda=l,)); paths[i,:].=_coefficients(Tilia.fit(changed,X,y)); end
    feature_names===nothing && (feature_names=["x$(i)" for i in axes(X,2)]); fig=Makie.Figure(size=(760,520)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Regularization λ",ylabel="Coefficient",xscale=log10)
    for j in axes(paths,2); Makie.lines!(ax,lambdas,paths[:,j];color=_palette(j),label=string(feature_names[j])); end; Makie.hlines!(ax,[0];color=:gray60); Makie.axislegend(ax;position=:rb); fig
end

"""Display Gaussian-mixture probability-density contours and component ellipses."""
function mixturedensityplot(fitted,X;resolution=180,title="Gaussian-mixture density")
    X=_matrix2(X); xs,ys,grid=_grid2(X;resolution); density=zeros(size(grid,1))
    for k in axes(fitted.means,1); centered=grid.-permutedims(fitted.means[k,1:2]); q=vec(sum((centered*fitted.precisions[k][1:2,1:2]).*centered;dims=2)); density .+= fitted.mixture_weights[k].*exp.(-0.5.*(2*log(2pi)+fitted.log_determinants[k].+q)); end
    fig=Makie.Figure(size=(720,600)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Feature 1",ylabel="Feature 2"); Makie.contourf!(ax,xs,ys,reshape(density,length(xs),length(ys));levels=18,colormap=:magma); Makie.scatter!(ax,X[:,1],X[:,2];color=(:white,0.45),markersize=5)
    for k in axes(fitted.means,1); _ellipse!(ax,fitted.means[k,1:2],fitted.covariances[k][1:2,1:2];color=:white); end; fig
end

"""Connect query observations to the neighbors returned by a fitted nearest-neighbor model."""
function neighborhoodplot(fitted,queries; n_neighbors=5,title="Nearest-neighbor neighborhoods")
    train=_matrix2(fitted.training_data); queries=_matrix2(queries); distances,indices=Tilia.kneighbors(fitted,queries;n_neighbors=n_neighbors)
    fig=Makie.Figure(size=(720,600)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Feature 1",ylabel="Feature 2"); Makie.scatter!(ax,train[:,1],train[:,2];color=(:gray45,0.65),label="Training")
    for q in axes(queries,1), j in axes(indices,2); i=indices[q,j]; Makie.lines!(ax,[queries[q,1],train[i,1]],[queries[q,2],train[i,2]];color=(:steelblue,0.35)); end
    Makie.scatter!(ax,queries[:,1],queries[:,2];marker=:star5,color=:darkorange,markersize=22,label="Query"); Makie.axislegend(ax); fig
end

"""Plot the anomaly-score distribution and the fitted isolation-forest cutoff."""
function anomalyscoreplot(fitted,X;title="Isolation-forest anomaly scores")
    scores=Tilia.anomaly_score(fitted,X); fig=Makie.Figure(size=(720,500)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Anomaly score",ylabel="Count"); Makie.hist!(ax,scores;bins=25,color=:steelblue); Makie.vlines!(ax,[fitted.threshold];color=:crimson,linewidth=3,label="Threshold"); Makie.axislegend(ax); fig
end

_trialparam(t,name) = getproperty(t.parameters,Symbol(name))
"""Heatmap two tuning parameters using the scores stored in a tuning result."""
function tuningheatmap(result; xparameter,yparameter,title="Tuning landscape")
    xs=unique([_trialparam(t,xparameter) for t in result.trials]); ys=unique([_trialparam(t,yparameter) for t in result.trials]); z=fill(NaN,length(xs),length(ys))
    for t in result.trials; z[findfirst(==(_trialparam(t,xparameter)),xs),findfirst(==(_trialparam(t,yparameter)),ys)]=t.score; end
    fig=Makie.Figure(size=(700,560)); ax=Makie.Axis(fig[1,1],title=title,xlabel=string(xparameter),ylabel=string(yparameter),xticks=(1:length(xs),string.(xs)),yticks=(1:length(ys),string.(ys))); hm=Makie.heatmap!(ax,1:length(xs),1:length(ys),z;colormap=:viridis); Makie.Colorbar(fig[1,2],hm,label="Score"); fig
end

"""Parallel-coordinate view of every hyperparameter trial, colored by score."""
function tuningparallelplot(result; parameters=nothing,title="Tuning trials")
    parameters===nothing && (parameters=collect(keys(first(result.trials).parameters))); values=[[getproperty(t.parameters,p) for t in result.trials] for p in parameters]; encoded=[v[1] isa Number ? Float64.(v) : Float64.(_encode(v)) for v in values]; normalized=[maximum(v)==minimum(v) ? zeros(length(v)) : (v.-minimum(v))./(maximum(v)-minimum(v)) for v in encoded]
    scores=[t.score for t in result.trials]; fig=Makie.Figure(size=(820,520)); ax=Makie.Axis(fig[1,1],title=title,xticks=(1:length(parameters),string.(parameters)),ylabel="Normalized value")
    for i in eachindex(scores); Makie.lines!(ax,1:length(parameters),[v[i] for v in normalized];color=scores[i],colormap=:viridis,colorrange=extrema(scores)); end; fig
end

"""Compare fold-score distributions from several cross-validation results."""
function modelcomparisonplot(results; names=["Model $(i)" for i in eachindex(results)],title="Model comparison")
    fig=Makie.Figure(size=(760,520)); ax=Makie.Axis(fig[1,1],title=title,xlabel="Model",ylabel="Cross-validation score",xticks=(1:length(results),string.(names)))
    for (i,r) in enumerate(results); jitter=range(-0.12,0.12;length=length(r.scores)); Makie.scatter!(ax,i .+ jitter,r.scores;color=_palette(i)); m=_mean(r.scores); s=_standard_deviation(r.scores); Makie.errorbars!(ax,[i],[m],[s];color=:black,whiskerwidth=14); Makie.scatter!(ax,[i],[m];marker=:diamond,color=:white,strokecolor=:black,markersize=15); end; fig
end
