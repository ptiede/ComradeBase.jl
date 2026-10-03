module ComradeBasePolyesterExt
using ComradeBase
using Polyester: @batch

const PolyThreads = ComradeBase.ThreadsEx{:Polyester}

function ComradeBase._threads_intensitymap!(
        img::IntensityMap,
        s::ComradeBase.AbstractModel, g,
        ::Val{:Polyester}
    )
    dx, dy = ComradeBase.pixelsizes(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = baseimage(img)
    cis = ComradeBase._pointindices(pimg, g)
    @batch for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        ComradeBase._setpoint!(pimg, I, f(g[I]) * dx * dy)
    end
    return nothing
end

function ComradeBase._threads_pointmap!(dest, f, g, ::Val{:Polyester})
    cis = ComradeBase._pointindices(dest, g)
    @batch for i in eachindex(IndexLinear(), cis)
        I = cis[i]
        ComradeBase._setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

end
