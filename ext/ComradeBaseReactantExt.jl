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
    ) where {T <: StokesParams, D}

    arrs = StructArrays.buildfromschema(x -> similar(Reactant.TracedRArray{unwrapped_eltype(x)}, size(g)), T)
    return IntensityMap(arrs, g)
end

function ComradeBase.domainpoints(d::RectiGrid{D, <:ComradeBase.ReactantEx}) where {D}
    g = map(Reactant.materialize_traced_array ∘ basedim, named_dims(d))
    rot = rotmat(d)
    return ComradeBase.LazyGrid(g, rot)
end

struct NamedIT{K, M, R}
    s::M
    rm::R
end

@inline function img_point(n::NamedIT{K}, ps...) where {K}
    psnr = ComradeBase.apply_transform(n.rm, ps)
    return ComradeBase.intensity_point(n.s, NamedTuple{K}(psnr))
end


function ComradeBase.intensitymap_analytic_executor!(
        img::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}
    dx, dy = pixelsizes(img)
    dms = map(Reactant.materialize_traced_array ∘ ComradeBase.basedim, named_dims(img))
    ddims = shapedims(values(dms))
    K = keys(dms)
    itp = NamedIT{K, typeof(s), typeof(rotmat(axisdims(img)))}(s, rotmat(axisdims(img)))
    bimg = baseimage(img)
    bimg .= img_point.(Ref(itp), ddims...) .* dx .* dy
    return nothing
end

@inline function vis_point(n::NamedIT{K}, ps...) where {K}
    psnr = ComradeBase.apply_transform(n.rm, ps)
    return ComradeBase.visibility_point(n.s, NamedTuple{K}(psnr))
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}

    dms = map(Reactant.materialize_traced_array ∘ ComradeBase.basedim, named_dims(vis))
    ddims = shapedims(values(dms))
    K = keys(dms)
    itp = NamedIT{K, typeof(s), typeof(rotmat(axisdims(vis)))}(s, rotmat(axisdims(vis)))
    bvis = baseimage(vis)
    bvis .= vis_point.(Ref(itp), ddims...)
    return nothing
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
    bvis = baseimage(vis)
    fa = Base.Fix1(ComradeBase.visibility_point, s)
    bvis .= fa.(g)
    return nothing
end


end
