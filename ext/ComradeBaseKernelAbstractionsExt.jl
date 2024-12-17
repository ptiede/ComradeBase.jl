module ComradeBaseKernelAbstractionsExt

using ComradeBase
using KernelAbstractions: Backend, allocate
using StructArrays

function ComradeBase.allocate_map(::Type{<:AbstractArray{T}},
                                  g::UnstructuredDomain{D,<:Backend}) where {T,D}
    return ComradeBase.UnstructuredMap(allocate(executor(g), T, size(g)), g)
end

function ComradeBase.allocate_map(::Type{<:StructArray{T}},
                                  g::UnstructuredDomain{D,<:Backend}) where {T,D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
    return UnstructuredMap(arrs, g)
end

function ComradeBase.allocate_map(::Type{<:AbstractArray{T}},
                                  g::ComradeBase.AbstractRectiGrid{D,<:Backend}) where {T,D}
    exec = executor(g)
    return IntensityMap(allocate(exec, T, size(g)), g)
end

function ComradeBase.allocate_map(::Type{<:StructArray{T}},
                                  g::ComradeBase.AbstractRectiGrid{D,<:Backend}) where {T,D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
    return IntensityMap(arrs, g)
end

function ComradeBase.intensitymap_analytic_executor!(img::IntensityMap,
                                                     s::ComradeBase.AbstractModel,
                                                     ::Backend)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    bimg .= intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function ComradeBase.intensitymap_analytic_executor!(img::UnstructuredMap,
                                                     s::ComradeBase.AbstractModel,
                                                     ::Backend)
    g = domainpoints(img)
    pvis = baseimage(vis)
    pvis .= intensity_point.(Ref(s), g)
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(vis::ComradeBase.FluxMap2,
                                                      s::ComradeBase.AbstractModel,
                                                      ::Backend)
    g = domainpoints(vis)
    pvis = baseimage(vis)
    pvis .= visibility_point.(Ref(s), g)
    return nothing
end

end
