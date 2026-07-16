using Documenter
using Tilia

makedocs(
    modules=[Tilia],
    sitename="Tilia.jl",
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, repolink=nothing),
    remotes=nothing,
    checkdocs=:exports,
    warnonly=[:cross_references],
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Data and schemas" => "data-and-schemas.md",
        "Pipelines and graphs" => "pipelines-and-graphs.md",
        "Models" => "models.md",
        "Model numerical contracts" => "model-semantics.md",
        "Metrics" => "metrics.md",
        "Model selection" => "model-selection.md",
        "Visualization with Makie" => "visualization.md",
        "Acceleration" => "acceleration.md",
        "Differentiation" => "differentiation.md",
        "Persistence" => "persistence.md",
        "Numerical behavior" => "numerical-behavior.md",
        "Extending Tilia" => "extending.md",
        "Internals" => "internals.md",
    ],
)
