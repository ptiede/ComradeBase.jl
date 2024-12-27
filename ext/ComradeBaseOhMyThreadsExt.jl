module ComradeBaseOhMyThreadsExt

using ComradeBase
using OhMyThreads

function ComradeBase.intensitymap_analytic_executor!(img::IntensityMap,
                                                     s::ComradeBase.AbstractModel,
                                                     executor::OhMyThreads.Scheduler)
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

function ComradeBase.intensitymap_analytic_executor!(img::UnstructuredMap,
                                                     s::ComradeBase.AbstractModel,
                                                     executor::OhMyThreads.Scheduler)
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @tasks for I in eachindex(pimg, g)
        @set scheduler = executor
        @inbounds pimg[I] = f(g[I])
    end
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(vis::ComradeBase.FluxMap2,
                                                      s::ComradeBase.AbstractModel,
                                                      executor::OhMyThreads.Scheduler)
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @tasks for I in eachindex(pimg, g)
        @set scheduler = executor
        @inbounds pimg[I] = f(g[I])
    end
    return nothing
end

end
