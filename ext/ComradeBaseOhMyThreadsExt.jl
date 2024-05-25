module ComradeBaseOhMyThreadsExt

using ComradeBase
using OhMyThreads

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.AbstractRectiGrid{D, <:OhMyThreads.Scheduler}) where {D}
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = tmap(g; scheduler=executor(dims)) do p
        f(p)*dx*dy
    end
    return img
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,<:ComradeBase.AbstractRectiGrid{D, <:OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,N,D}
    dims = axisdims(img)
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    tforeach(CartesianIndices(img); scheduler=executor(dims)) do I
        img[I] = f(g[I])*dx*dy
    end
    return nothing
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.UnstructuredDomain{D, <:OhMyThreads.Scheduler}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = tmap(f, g; scheduler=executor(dims))
    return img
end


function ComradeBase.intensitymap_analytic!(img::UnstructuredMap{T,<:AbstractVector,<:UnstructuredDomain{D, <:OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,D}
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    tforeach(CartesianIndices(img); scheduler=executor(dims)) do I
        img[I] = f(g[I])
    end
    return nothing
end

function ComradeBase.visibilitymap_analytic(m::ComradeBase.AbstractModel, dims::ComradeBase.AbstractSingleDomain{D, <:OhMyThreads.Scheduler}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, m)
    img = tmap(f, g; scheduler=executor(dims))
    return img
end

function ComradeBase.visibilitymap_analytic!(vis::ComradeBase.FluxMap2{T,N,<:ComradeBase.AbstractSingleDomain{<:Any, <: OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,N}
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    tforeach(CartesianIndices(vis); scheduler=executor(dims)) do I
        vis[I] = f(g[I])
    end
    return nothing
end

end
