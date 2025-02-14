module StokedBaseOhMyThreadsExt

using StokedBase
using OhMyThreads

function StokedBase.intensitymap_analytic_executor!(img::IntensityMap,
                                                    s::StokedBase.AbstractModel,
                                                    executor::OhMyThreads.Scheduler)
    dims = axisdims(img)
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    pimg = parent(img)
    f = Base.Fix1(StokedBase.intensity_point, s)

    # TODO: Open issue on OhMyThreads to support CartesianIndices
    @tasks for I in eachindex(pimg, g)
        @set scheduler = executor
        @inbounds pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function StokedBase.intensitymap_analytic_executor!(img::UnstructuredMap,
                                                    s::StokedBase.AbstractModel,
                                                    executor::OhMyThreads.Scheduler)
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(StokedBase.intensity_point, s)
    pimg = parent(img)
    @tasks for I in eachindex(pimg, g)
        @set scheduler = executor
        @inbounds pimg[I] = f(g[I])
    end
    return nothing
end

function StokedBase.visibilitymap_analytic_executor!(vis::StokedBase.FluxMap2,
                                                     s::StokedBase.AbstractModel,
                                                     executor::OhMyThreads.Scheduler)
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(StokedBase.visibility_point, s)
    pvis = parent(vis)
    @tasks for I in eachindex(pvis, g)
        @set scheduler = executor
        @inbounds pvis[I] = f(g[I])
    end
    return nothing
end

end
