push!(LOAD_PATH,"../src/")
using Documenter, StateSpace

makedocs(
    sitename="StateSpace.jl",
    modules=[StateSpace],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)