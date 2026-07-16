ENV["JULIA_NUM_PRECOMPILE_TASKS"] = 1
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CairoMakie
using DataFrames
using MLDatasets
using Tilia
using TiliaMakieRecipes

const OUTPUT_DIRECTORY = joinpath(@__DIR__,"output","explanatory")
mkpath(OUTPUT_DIRECTORY)

iris=Iris(as_df=false)
X=Float64.(Matrix(transpose(iris.features)))
y=replace.(vec(iris.targets),"Iris-"=>"")
names=["Sepal length","Sepal width","Petal length","Petal width"]

classifier=fit(LogisticRegression(max_iterations=500),X,y)
regressor=fit(HistGradientBoostingRegressor(n_estimators=100,max_depth=4),X[:,1:3],X[:,4])

figures=(
    ("regression_partial_dependence_ice.png",partialdependenceplot(regressor,X[:,1:3];feature=3,resolution=45,observations=35,feature_names=names[1:3],title="Petal length effect · PDP and ICE")),
    ("classification_partial_dependence_contour.png",partialdependenceplot(classifier,X;feature=(3,4),target="virginica",resolution=36,feature_names=names,title="Virginica partial dependence")),
    ("classification_partial_dependence_surface.png",partialdependenceplot(classifier,X;feature=(3,4),target="virginica",resolution=32,feature_names=names,surface=true,title="Virginica probability surface")),
    ("iris_probability_simplex.png",probabilitysimplexplot(classifier,X;groups=y,title="Iris logistic probability simplex",markersize=11)),
)

for (filename,figure) in figures
    save(joinpath(OUTPUT_DIRECTORY,filename),figure;px_per_unit=1.8)
end

println("Generated ",length(figures)," explanatory figures in: ",OUTPUT_DIRECTORY)
