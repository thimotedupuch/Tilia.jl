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
        "User guide" => [
            "Data and schemas" => "data-and-schemas.md",
            "Pipelines and graphs" => "pipelines-and-graphs.md",
            "Models" => "models.md",
            "Metrics and inspection" => "metrics.md",
            "Model selection" => "model-selection.md",
        ],
        "Operations" => [
            "Numerical behavior" => "numerical-behavior.md",
            "Persistence" => "persistence.md",
        ],
        "Optional integrations" => [
            "Visualization with Makie" => "visualization.md",
            "Acceleration" => "acceleration.md",
            "Differentiation" => "differentiation.md",
        ],
        "Developer guide" => [
            "Extending Tilia" => "extending.md",
            "Model numerical contracts" => "model-semantics.md",
            "Internals and API reference" => "internals.md",
        ],
    ],
)



deploydocs(
  repo="github.com/thimotedupuch/Tilia.jl.git",
  devbranch="master",
)
