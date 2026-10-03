module ComradeBaseKernelAbstractionsExt

using ComradeBase
using KernelAbstractions: Backend, allocate
using StructArrays

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}},
        g::ComradeBase.AbstractRectiGrid{D, <:Backend}
    ) where {T, D}
    return _allocate_backend_map(T, g)
end

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}},
        g::ComradeBase.StructuredDomain{<:Tuple, <:NamedTuple, <:NamedTuple, <:Backend}
    ) where {T}
    return _allocate_backend_map(T, g)
end

function ComradeBase.allocate_map(
        ::Type{<:StructArray{T}},
        g::ComradeBase.AbstractRectiGrid{D, <:Backend}
    ) where {T, D}
    return _allocate_backend_structmap(T, g)
end

function ComradeBase.allocate_map(
        ::Type{<:StructArray{T}},
        g::ComradeBase.StructuredDomain{<:Tuple, <:NamedTuple, <:NamedTuple, <:Backend}
    ) where {T}
    return _allocate_backend_structmap(T, g)
end

_allocate_backend_map(T, g) = IntensityMap(allocate(executor(g), T, size(g)), g)

function _allocate_backend_structmap(T, g)
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
    return IntensityMap(arrs, g)
end

function ComradeBase.intensitymap_analytic_executor!(
        img::ComradeBase.RectiMap,
        s::ComradeBase.AbstractModel,
        ::Backend
    )
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    bimg .= ComradeBase.intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function ComradeBase._pointmap!(dest, f, d, ::Backend)
    return ComradeBase._broadcast_pointmap!(dest, f, d)
end

end
