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
    pimg = parent(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)

    # TODO: Open issue on OhMyThreads to support CartesianIndices
    @tasks for I in eachindex(pimg, g)
        @set scheduler = executor
        @inbounds pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function ComradeBase._pointmap!(dest, f, d, executor::OhMyThreads.Scheduler)
    g = domainpoints(d)
    @tasks for I in eachindex(dest, g)
        @set scheduler = executor
        dest[I] = f(g[I])
    end
    return nothing
end

end
