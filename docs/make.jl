using Baytes
using Documenter

DocMeta.setdocmeta!(Baytes, :DocTestSetup, :(using Baytes); recursive=true)

makedocs(;
    modules=[Baytes],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/Baytes.jl/blob/{commit}{path}#{line}",
    sitename="Baytes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/Baytes.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/Baytes.jl",
    devbranch="main",
)
