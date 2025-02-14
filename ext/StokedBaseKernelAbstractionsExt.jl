module StokedBaseKernelAbstractionsExt

using StokedBase
using KernelAbstractions: Backend, allocate
using StructArrays

function StokedBase.allocate_map(::Type{<:AbstractArray{T}},
                                 g::UnstructuredDomain{D,<:Backend}) where {T,D}
    return StokedBase.UnstructuredMap(allocate(executor(g), T, size(g)), g)
end

function StokedBase.allocate_map(::Type{<:StructArray{T}},
                                 g::UnstructuredDomain{D,<:Backend}) where {T,D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
    return UnstructuredMap(arrs, g)
end

function StokedBase.allocate_map(::Type{<:AbstractArray{T}},
                                 g::StokedBase.AbstractRectiGrid{D,<:Backend}) where {T,D}
    exec = executor(g)
    return IntensityMap(allocate(exec, T, size(g)), g)
end

function StokedBase.allocate_map(::Type{<:StructArray{T}},
                                 g::StokedBase.AbstractRectiGrid{D,<:Backend}) where {T,D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
    return IntensityMap(arrs, g)
end

function StokedBase.intensitymap_analytic_executor!(img::IntensityMap,
                                                    s::StokedBase.AbstractModel,
                                                    ::Backend)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    bimg .= StokedBase.intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function StokedBase.intensitymap_analytic_executor!(img::UnstructuredMap,
                                                    s::StokedBase.AbstractModel,
                                                    ::Backend)
    g = domainpoints(img)
    pvis = baseimage(vis)
    pvis .= StokedBase.intensity_point.(Ref(s), g)
    return nothing
end

function StokedBase.visibilitymap_analytic_executor!(vis::StokedBase.FluxMap2,
                                                     s::StokedBase.AbstractModel,
                                                     ::Backend)
    g = domainpoints(vis)
    pvis = baseimage(vis)
    pvis .= StokedBase.visibility_point.(Ref(s), g)
    return nothing
end

end
