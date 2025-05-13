using ComradeBase
using Documenter

DocMeta.setdocmeta!(ComradeBase, :DocTestSetup, :(using ComradeBase); recursive = true)

makedocs(;
    modules = [ComradeBase],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    repo = "https://github.com/ptiede/ComradeBase.jl/blob/{commit}{path}#{line}",
    sitename = "ComradeBase.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ptiede.github.io/ComradeBase.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/ptiede/ComradeBase.jl",
    devbranch = "main",
)
