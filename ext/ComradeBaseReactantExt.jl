module ComradeBaseReactantExt

using ComradeBase
using StructArrays
using Reactant
using StaticArrays

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

# function ComradeBase.allocate_map(
#         ::Type{<:AbstractArray{T}},
#         g::UnstructuredDomain{D, <:ReactantEx}
#     ) where {T, D}
#     result = UnstructuredMap(similar(TracedRArray{unwrapped_eltype(T)}, size(g)), g)
#     return result
# end

# function ComradeBase.allocate_map(
#         ::Type{<:AbstractArray{Reactant.TracedRNumber{T}}},
#         g::ComradeBase.AbstractRectiGrid
#     ) where {T}
#     arr = similar(Reactant.TracedRArray{T}, size(g))
#     return IntensityMap(arr, g)
# end

# function ComradeBase.allocate_map(
#         ::Type{<:AbstractArray{T}},
#         g::ComradeBase.AbstractRectiGrid{D, <:ReactantEx}
#     ) where {T, D}
#     arr = similar(ConcreteRArray{T}, size(g))
#     return IntensityMap(arr, g)
# end

@inline function ComradeBase.similartype(::IsPolarized, ::Type{<:ReactantEx}, ::Type{T}) where {T}
    return StructArray{StokesParams{Reactant.TracedRNumber{unwrapped_eltype(T)}}}
end

@inline function ComradeBase.similartype(::NotPolarized, ::Type{<:ReactantEx}, ::Type{T}) where {T}
    return TracedRArray{unwrapped_eltype(T)}
end




# Copied from ComradeBaseKernelAbstractionsExt, these
# probably will need to be modified still:

function ComradeBase.allocate_map(
        ::Type{<:StructArray{T}},
        g::ComradeBase.AbstractRectiGrid{D, <:ReactantEx}
    ) where {T<:StokesParams, D}
    
    arrs = StructArrays.buildfromschema(x -> similar(Reactant.TracedRArray{unwrapped_eltype(x)}, size(g)), T)
    return IntensityMap(arrs, g)
end

function ComradeBase.domainpoints(d::RectiGrid{D, <:ComradeBase.ReactantEx}) where {D}
    g = Reactant.materialize_traced_array.(map(basedim, dims(d)))
    rot = rotmat(d)
    N = keys(d)
    return ComradeBase.LazyGrid()
end


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

function img_point(p, rm, K, ps...)
    psnr = ComradeBase.apply_transform(rm, ps)
    return ComradeBase.intensity_point(p, NamedTuple{K}(psnr))
end

function ComradeBase.intensitymap_analytic_executor!(
        img::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}
    dx, dy = pixelsizes(img)
    dms = Reactant.materialize_traced_array.(map(basedim, dims(img)))
    ddims = ntuple(k -> reshape(dms[k], ntuple(i -> i == k ? Base.Colon() : 1, Val(N))), Val(N))
    rm = rotmat(axisdims(img))
    bimg = baseimage(img)
    K = keys(axisdims(img))
    bimg .= img_point.(Ref(s), Ref(rm), Ref(K), ddims...) .* dx .* dy
    return nothing
end

function vis_point(p, rm, K, ps...)
    psnr = ComradeBase.apply_transform(rm, ps)
    return ComradeBase.visibility_point(p, NamedTuple{K}(psnr))
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}
    dms = Reactant.materialize_traced_array.(map(basedim, dims(vis)))
    ddims = ntuple(k -> reshape(dms[k], ntuple(i -> i == k ? Base.Colon() : 1, Val(N))), Val(N))
    rm = rotmat(axisdims(vis))
    bvis = baseimage(vis)
    K = keys(axisdims(vis))
    bvis .= vis_point.(Ref(s), Ref(rm), Ref(K), ddims...)
    # return nothing
end


function ComradeBase.intensitymap_analytic_executor!(
        img::UnstructuredMap,
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    )
    g = domainpoints(img)
    bimg = baseimage(img)
    fa = Base.Fix1(ComradeBase.intensity_point, s)
    bimg .= fa.(g)
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::UnstructuredMap,
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    )
    g = domainpoints(vis)
    bimg = baseimage(vis)
    fa = Base.Fix1(ComradeBase.visibility_point, s)
    bimg .= fa.(g)
    return nothing
end



end
