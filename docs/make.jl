using StokedBase
using Documenter

DocMeta.setdocmeta!(StokedBase, :DocTestSetup, :(using StokedBase); recursive=true)

makedocs(;
         modules=[StokedBase],
         authors="Paul Tiede <ptiede91@gmail.com> and contributors",
         repo="https://github.com/ptiede/StokedBase.jl/blob/{commit}{path}#{line}",
         sitename="StokedBase.jl",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://ptiede.github.io/StokedBase.jl",
                                assets=String[],),
         pages=["Home" => "index.md"],)

deploydocs(;
           repo="github.com/ptiede/StokedBase.jl",
           devbranch="main",)
