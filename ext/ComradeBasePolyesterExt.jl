module ComradeBasePolyesterExt
using ComradeBase
using Polyester: @batch

function ComradeBase._threads_pointmap!(dest, f, g, ::Val{:Polyester})
    cis = ComradeBase._pointindices(dest, g)
    @batch for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        dest[I] = f(g[I])
    end
    return nothing
end

end
