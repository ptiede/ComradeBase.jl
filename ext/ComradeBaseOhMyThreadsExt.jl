module ComradeBaseOhMyThreadsExt

using ComradeBase
using OhMyThreads

function ComradeBase.intensitymap_analytic_executor!(
        img::ComradeBase.RectiMap,
        s::ComradeBase.AbstractModel,
        executor::OhMyThreads.Scheduler
    )
    dims = axisdims(img)
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    pimg = baseimage(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    cis = ComradeBase._pointindices(pimg, g)

    # TODO: Open issue on OhMyThreads to support CartesianIndices
    @tasks for i in eachindex(IndexLinear(), cis)
        @set scheduler = executor
        I = cis[i]
        ComradeBase._setpoint!(pimg, I, f(g[I]) * dx * dy)
    end
    return nothing
end

function ComradeBase._pointmap!(dest, f, d, executor::OhMyThreads.Scheduler)
    g = domainpoints(d)
    cis = ComradeBase._pointindices(dest, g)
    @tasks for i in eachindex(IndexLinear(), cis)
        @set scheduler = executor
        I = cis[i]
        ComradeBase._setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

end
