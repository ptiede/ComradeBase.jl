module ComradeBaseReactantExt

using ComradeBase
using StructArrays
using Reactant

import ComradeBase: AbstractSingleDomain, basedim, dims, UnstructuredMap
import Reactant: TracedRArray, unwrapped_eltype

struct ReactantBackend
end

Base.eltype(d::AbstractSingleDomain{D, E}) where {D, E <: ReactantBackend} = Reactant.allowscalar() do
    eltype(basedim(first(dims(d))))
end

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}},
        g::UnstructuredDomain{D, <:ReactantBackend}
    ) where {T, D}
    result = UnstructuredMap(similar(TracedRArray{unwrapped_eltype(T)}, size(g)), g)
    return result
end

# Copied from ComradeBaseKernelAbstractionsExt, these
# probably will need to be modified still:

# function ComradeBase.allocate_map(
#         ::Type{<:StructArray{T}},
#         g::UnstructuredDomain{D, <:ReactantBackend}
#     ) where {T, D}
#     exec = executor(g)
#     arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
#     return UnstructuredMap(arrs, g)
# end

# function ComradeBase.allocate_map(
#         ::Type{<:AbstractArray{T}},
#         g::ComradeBase.AbstractRectiGrid{D, <:ReactantBackend}
#     ) where {T, D}
#     exec = executor(g)
#     return IntensityMap(allocate(exec, T, size(g)), g)
# end

# function ComradeBase.allocate_map(
#         ::Type{<:StructArray{T}},
#         g::ComradeBase.AbstractRectiGrid{D, <:ReactantBackend}
#     ) where {T, D}
#     exec = executor(g)
#     arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
#     return IntensityMap(arrs, g)
# end

function ComradeBase.intensitymap_analytic_executor!(
        img::IntensityMap,
        s::ComradeBase.AbstractModel,
        ::ReactantBackend
    )
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    bimg .= ComradeBase.intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function ComradeBase.intensitymap_analytic_executor!(
        img::UnstructuredMap,
        s::ComradeBase.AbstractModel,
        ::ReactantBackend
    )
    g = domainpoints(img)
    pvis = baseimage(vis)
    pvis .= ComradeBase.intensity_point.(Ref(s), g)
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::ComradeBase.FluxMap2,
        s::ComradeBase.AbstractModel,
        ::ReactantBackend
    )
    g = domainpoints(vis)
    pvis = baseimage(vis)
    pvis .= ComradeBase.visibility_point.(Ref(s), g)
    return nothing
end

end
