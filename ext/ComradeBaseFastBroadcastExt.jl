module ComradeBaseFastBroadcastExt

using ComradeBase
using FastBroadcast

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.AbstractRectiGrid{D, <:ComradeBase.FastBroadEx{false}}) where {D}
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = @.. f(g)*dx*dy
    return IntensityMap(img, dims)
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,<:ComradeBase.AbstractRectiGrid{D, <:ComradeBase.FastBroadEx{false}}}, s::ComradeBase.AbstractModel) where {T,N,D}
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    @.. img = f(g)*dx*dy
    return nothing
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.AbstractRectiGrid{D, <:ComradeBase.FastBroadEx{true}}) where {D}
    dx = step(dims.X)
    dy = step(dims.Y)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = @.. thread=true f(g)*dx*dy
    return IntensityMap(img, dims)
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,<:ComradeBase.AbstractRectiGrid{D, <:ComradeBase.FastBroadEx{true}}}, s::ComradeBase.AbstractModel) where {T,N,D}
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    @.. thread=true img = f(g)*dx*dy
    return nothing
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.UnstructuredDomain{D, <:ComradeBase.FastBroadEx{false}}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = @.. f(g)
    return img
end


function ComradeBase.intensitymap_analytic!(img::UnstructuredMap{T,<:AbstractVector,<:UnstructuredDomain{D, <:ComradeBase.FastBroadEx{false}}}, s::ComradeBase.AbstractModel) where {T,D}
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @.. pimg = f(g)
    return nothing
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.UnstructuredDomain{D, <:ComradeBase.FastBroadEx{true}}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = @.. thread=true f(g)
    return img
end


function ComradeBase.intensitymap_analytic!(img::UnstructuredMap{T,<:AbstractVector,<:UnstructuredDomain{D, <:ComradeBase.FastBroadEx{true}}}, s::ComradeBase.AbstractModel) where {T,D}
    dims = axisdims(img)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @.. thread=true pimg = f(g)
    return nothing
end

function ComradeBase.visibilitymap_analytic(m::ComradeBase.AbstractModel, dims::ComradeBase.AbstractSingleDomain{D, <:ComradeBase.FastBroadEx{false}}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, m)
    img = @.. f(g)
    return img
end

function ComradeBase.visibilitymap_analytic!(vis::ComradeBase.FluxMap2{T,N,<:ComradeBase.AbstractSingleDomain{<:Any, <: ComradeBase.FastBroadEx{false}}}, s::ComradeBase.AbstractModel) where {T,N}
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @.. pvis = f(g)
    return nothing
end

function ComradeBase.visibilitymap_analytic(m::ComradeBase.AbstractModel, dims::ComradeBase.AbstractSingleDomain{D, <:ComradeBase.FastBroadEx{true}}) where {D}
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, m)
    img = @.. thread=true f(g)
    return img
end

function ComradeBase.visibilitymap_analytic!(vis::ComradeBase.FluxMap2{T,N,<:ComradeBase.AbstractSingleDomain{<:Any, <: ComradeBase.FastBroadEx{true}}}, s::ComradeBase.AbstractModel) where {T,N}
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @.. thread=true pvis = f(g)
    return nothing
end



end
