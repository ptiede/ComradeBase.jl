module ComradeBaseReactantExt

using ComradeBase
using StructArrays
using Reactant

import ComradeBase: AbstractSingleDomain, basedim, dims, UnstructuredMap
using ComradeBase: ReactantEx
import Reactant: AnyTracedRArray, TracedRArray, unwrapped_eltype

const RInt = Union{Integer, Reactant.TracedRNumber{<:Integer}}

Base.@propagate_inbounds function ComradeBase.rgetindex(I::Reactant.AnyTracedRArray, i::RInt...)
    return @allowscalar I[i...]
end

Base.@propagate_inbounds function ComradeBase.rsetindex!(I::Reactant.AnyTracedRArray, v, i::RInt...)
    return @allowscalar I[i...] = v
end


# If inside tracing land we automatically switch the backend to Reactant
Base.@nospecializeinfer function Reactant.make_tracer(
        seen,
        @nospecialize(prev::Union{ComradeBase.Serial, ComradeBase.ThreadsEx}),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        @nospecialize(sharding = Reactant.Sharding.NoSharding()),
        @nospecialize(runtime),
        kwargs...
    )
    return Reactant.traced_type(typeof(prev), Val(mode), track_numbers, sharding, runtime)()
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:Union{ComradeBase.Serial, ComradeBase.ThreadsEx}}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    return ReactantEx
end


Base.eltype(d::AbstractSingleDomain{D, E}) where {D, E <: ReactantEx} = Reactant.allowscalar() do
    eltype(basedim(first(dims(d))))
end

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}},
        g::UnstructuredDomain{D, <:ReactantEx}
    ) where {T, D}
    result = UnstructuredMap(similar(TracedRArray{unwrapped_eltype(T)}, size(g)), g)
    return result
end

# Copied from ComradeBaseKernelAbstractionsExt, these
# probably will need to be modified still:

# function ComradeBase.allocate_map(
#         ::Type{<:StructArray{T}},
#         g::UnstructuredDomain{D, <:ReactantEx}
#     ) where {T, D}
#     exec = executor(g)
#     arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
#     return UnstructuredMap(arrs, g)
# end

# function ComradeBase.allocate_map(
#         ::Type{<:AbstractArray{T}},
#         g::ComradeBase.AbstractRectiGrid{D, <:ReactantEx}
#     ) where {T, D}
#     exec = executor(g)
#     return IntensityMap(allocate(exec, T, size(g)), g)
# end

# function ComradeBase.allocate_map(
#         ::Type{<:StructArray{T}},
#         g::ComradeBase.AbstractRectiGrid{D, <:ReactantEx}
#     ) where {T, D}
#     exec = executor(g)
#     arrs = StructArrays.buildfromschema(x -> allocate(exec, x, size(g)), T)
#     return IntensityMap(arrs, g)
# end

function ComradeBase.intensitymap_analytic_executor!(
        img::IntensityMap,
        s::ComradeBase.AbstractModel,
        ::ReactantEx
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
        ::ReactantEx
    )
    g = domainpoints(img)
    pvis = baseimage(vis)
    pvis .= ComradeBase.intensity_point.(Ref(s), g)
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::ComradeBase.FluxMap2,
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    )
    g = domainpoints(vis)
    pvis = baseimage(vis)
    pvis .= ComradeBase.visibility_point.(Ref(s), g)
    return nothing
end

end
