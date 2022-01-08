using ROSEBase
using Documenter

DocMeta.setdocmeta!(ROSEBase, :DocTestSetup, :(using ROSEBase); recursive=true)

makedocs(;
    modules=[ROSEBase],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ROSEBase.jl/blob/{commit}{path}#{line}",
    sitename="ROSEBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ROSEBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ROSEBase.jl",
    devbranch="main",
)
