module ComradeBaseEnzymeExt

using ComradeBase
using Enzyme: @parallel

const EnzymeThreads = ComradeBase.ThreadsEx{:Enzyme}

function ComradeBase._threads_intensitymap!(
        img::IntensityMap,
        s::ComradeBase.AbstractModel, g,
        ::Val{:Enzyme}
    )
    dx, dy = ComradeBase.pixelsizes(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = baseimage(img)
    cis = ComradeBase._pointindices(pimg, g)
    @parallel for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        ComradeBase._setpoint!(pimg, I, f(g[I]) * dx * dy)
    end
    return nothing
end

function ComradeBase._threads_pointmap!(dest, f, g, ::Val{:Enzyme})
    cis = ComradeBase._pointindices(dest, g)
    @parallel for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        ComradeBase._setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

end
