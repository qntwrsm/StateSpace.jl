push!(LOAD_PATH,"../src/")
using Documenter, StateSpace

makedocs(
    sitename="StateSpace.jl",
    modules=[StateSpace],
    pages = [
        "index.md",
        "models.md",
        "filter.md",
        "smoother.md",
        "estimation.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)