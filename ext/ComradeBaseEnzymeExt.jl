module ComradeBaseEnzymeExt

using ComradeBase
using Enzyme: @parallel

function ComradeBase._threads_pointmap!(dest, f, g, ::Val{:Enzyme})
    cis = ComradeBase._pointindices(dest, g)
    @parallel for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        ComradeBase._setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

end
