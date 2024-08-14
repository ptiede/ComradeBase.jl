module ComradeBaseOhMyThreadsExt

using ComradeBase
using OhMyThreads

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


function ComradeBase.intensitymap_analytic!(img::UnstructuredMap{T,<:AbstractVector,<:UnstructuredDomain{D, <:OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,D}
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    tforeach(CartesianIndices(img); scheduler=executor(dims)) do I
        img[I] = f(g[I])
    end
    return nothing
end

function ComradeBase.visibilitymap_analytic!(vis::ComradeBase.FluxMap2{T,N,<:ComradeBase.AbstractSingleDomain{<:Any, <: OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,N}
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    tforeach(CartesianIndices(vis); scheduler=executor(dims)) do I
        pvis[I] = f(g[I])
    end
    return nothing
end

end
