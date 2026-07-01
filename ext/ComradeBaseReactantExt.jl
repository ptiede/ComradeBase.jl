module ComradeBaseReactantExt

using ComradeBase
using StructArrays
using Reactant
using StaticArrays

import ComradeBase: AbstractSingleDomain, basedim, dims, UnstructuredMap
using ComradeBase: ReactantEx, ShardSpec, UnstructuredDomain
import Reactant: AnyTracedRArray, TracedRArray, unwrapped_eltype


# --- Sharding ---------------------------------------------------------------------------------
# The `ShardSpec` carried on a `ReactantEx` executor is only a *declaration* of how the user wants
# things laid out. The actual sharding is applied with the public `Reactant.to_rarray` API at the
# input boundary (`to_sharded` below); ordinary Julia code in the executors / NUFFT then propagates
# the sharding through tracing. No internal Reactant ops are used.

function ComradeBase.shardmesh(dims::Vararg{Int}; names)
    return Reactant.Sharding.Mesh(reshape(Reactant.devices(), dims...), names)
end

# Map each semantic axis name to its array-dimension position for the object being sharded.
# An unstructured domain/map — and a bare per-visibility vector (a measurement / noise vector with no
# executor of its own) — is a flat list of points, so the per-point labels :Ti and :Fr both refer to
# the single data dimension (dim 1); sharding "over Ti/Fr" means sharding that flat axis after the data
# has been `regroup`ed by the same label. Restricted to `AbstractVector` so a higher-rank payload fails
# loudly (MethodError) instead of being silently split on dim 1. Rectilinear objects map their named
# dims (:X, :Y, optionally :Ti, :Fr) to positions in order.
_axispositions(::Union{UnstructuredDomain, UnstructuredMap, AbstractVector}) = (; Ti = 1, Fr = 1)
function _axispositions(x::ComradeBase.AbstractRectiGrid)
    ks = keys(x)
    return NamedTuple{ks}(ntuple(identity, length(ks)))
end
# An IntensityMap is an AbstractArray, so `keys` would give CartesianIndices; take the named dims
# from its grid instead (the pixel-array dimensions follow the grid's dim order).
_axispositions(x::IntensityMap) = _axispositions(ComradeBase.axisdims(x))

# Translate a `ShardSpec` + the object's axis layout into a public `Reactant.Sharding.DimsSharding`.
# `DimsSharding` shards the listed dims and replicates the rest, so one spec works across the
# different-rank arrays held inside a domain/map. Multiple semantic axes can land on the same array
# dimension (e.g. :Ti and :Fr both map to the flat dim 1), and a mesh-axis value may itself be a tuple
# — in either case that dimension is sharded across the tuple of all the mesh axes assigned to it.
function _dimssharding(spec::ShardSpec, x)
    pos = _axispositions(x)
    bydim = Tuple{Int, Symbol}[]
    for an in keys(spec.axes)
        haskey(pos, an) || continue
        v = spec.axes[an]
        for m in (v isa Tuple ? v : (v,))
            push!(bydim, (pos[an], m))
        end
    end
    sdims = sort!(unique(first.(bydim)))
    pspec = map(sdims) do d
        ax = [m for (dd, m) in bydim if dd == d]
        length(ax) == 1 ? ax[1] : Tuple(ax)
    end
    return Reactant.Sharding.DimsSharding(spec.mesh, Tuple(sdims), Tuple(pspec))
end

# Resolve the executor's sharding *declaration* into a concrete Reactant sharding for `x`.
#  - `ShardSpec`: the semantic convenience layer (axis names -> DimsSharding).
#  - a callable: the escape hatch — full Reactant power; gets `x`, returns any AbstractSharding.
#  - an AbstractSharding: used as-is.
_resolve_sharding(spec::ShardSpec, x) = _dimssharding(spec, x)
_resolve_sharding(sh::Reactant.Sharding.AbstractSharding, x) = sh
_resolve_sharding(f, x) = f(x)

# Core sharding step: move `x` onto the device for a resolved sharding *declaration* (`nothing` =
# unsharded). Both public entry points below funnel through here so the one- and two-argument forms,
# and every object type, can never drift apart.
_to_sharded(x, ::Nothing) = Reactant.to_rarray(x)
_to_sharded(x, decl) = Reactant.to_rarray(x; sharding = _resolve_sharding(decl, x))

# An image map's grid axes (`X`, `Y`, `Ti`, `Fr`) are tiny coordinate vectors that the multidomain
# NUFFT plan builder iterates on the *host*; only the pixel array is large enough to be worth sharding.
# So shard the value array per the declaration and leave the grid replicated on the host (turning the
# coordinate arrays into device arrays would make that host iteration a disallowed scalar index). The
# declaration is resolved against the *image* (for its X/Y/Ti/Fr dim positions) but applied to the bare
# pixel array; `axisdims(img)` keeps the original host grid.
_to_sharded(img::IntensityMap, ::Nothing) = IntensityMap(Reactant.to_rarray(baseimage(img)), axisdims(img))
function _to_sharded(img::IntensityMap, decl)
    vals = Reactant.to_rarray(baseimage(img); sharding = _resolve_sharding(decl, img))
    return IntensityMap(vals, axisdims(img))
end

# One-argument form: shard `x` per its own executor's declaration.
ComradeBase.to_sharded(x) = _to_sharded(x, ComradeBase.sharding(executor(x)))

# Two-argument form: shard `x` per the declaration carried by `ex` rather than by `x`'s own executor.
# This is how a likelihood's flat `measurement`/`noise` vectors — which carry no executor — and a
# visibility domain are placed on the same device blocks from a single `ReactantEx` passed at
# `prepare_device` time. An `IntensityMap` still keeps its grid on the host via the shared core above.
ComradeBase.to_sharded(x, ex::ReactantEx) = _to_sharded(x, ComradeBase.sharding(ex))

const RInt = Union{Integer, Reactant.TracedRNumber{<:Integer}}
const TInt = Reactant.TracedRNumber{<:Integer}

Base.@propagate_inbounds function ComradeBase.rgetindex(I::Reactant.AnyTracedRArray, i::RInt...)
    return @allowscalar I[i...]
end

Base.@propagate_inbounds function ComradeBase.rsetindex!(I::Reactant.AnyTracedRArray, v, i::RInt...)
    return @allowscalar I[i...] = v
end

# A *plain* Julia array indexed with a traced index (e.g. a static chain-time table read
# inside an `@trace` loop): Reactant's interpreter promotes the array to a traced
# constant automatically; it only needs the scalar-indexing opt-in. The plain-index
# methods above stay untouched so the CPU fast path never pays for `@allowscalar`.
# (The AnyTracedRArray+traced-index copies below break the vararg specificity ambiguity
# with the RInt methods above.)
Base.@propagate_inbounds function ComradeBase.rgetindex(I::AbstractArray, i::TInt...)
    return @allowscalar I[i...]
end

Base.@propagate_inbounds function ComradeBase.rgetindex(I::Reactant.AnyTracedRArray, i::TInt...)
    return @allowscalar I[i...]
end

Base.@propagate_inbounds function ComradeBase.rsetindex!(I::AbstractArray, v, i::TInt...)
    return @allowscalar I[i...] = v
end

Base.@propagate_inbounds function ComradeBase.rsetindex!(I::Reactant.AnyTracedRArray, v, i::TInt...)
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
    # Serial/ThreadsEx carry no sharding intent, so they convert to an unsharded ReactantEx.
    # A domain that was explicitly given `ReactantEx(spec)` keeps its spec via the generic
    # struct tracer (the Mesh is a leaf type), so no rule is needed for `ReactantEx` itself.
    return ReactantEx()
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:Union{ComradeBase.Serial, ComradeBase.ThreadsEx}}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    return ReactantEx{Nothing}
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

struct ApplyIT{K, M, R}
    s::M
    rm::R
end
function ApplyIT{K}(s, rm) where {K}
    return ApplyIT{K, typeof(s), typeof(rm)}(s, rm)
end

@inline function giterate(n::ApplyIT{K}, ps...) where {K}
    psnr = ComradeBase.apply_transform(n.rm, ps)
    return n.s(NamedTuple{K}(psnr))
end


function ComradeBase.intensitymap_analytic_executor!(
        img::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}
    dx, dy = pixelsizes(img)
    dms = map(Reactant.materialize_traced_array ∘ ComradeBase.basedim, named_dims(img))
    ddims = ComradeBase.shapedims(values(dms))
    K = keys(dms)
    itp = ApplyIT{K}(Base.Fix1(ComradeBase.intensity_point, s), rotmat(axisdims(img)))
    bimg = baseimage(img)
    bimg .= giterate.(Ref(itp), ddims...) .* dx .* dy
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(
        vis::IntensityMap{T, N},
        s::ComradeBase.AbstractModel,
        ::ReactantEx
    ) where {T, N}

    dms = map(Reactant.materialize_traced_array ∘ ComradeBase.basedim, named_dims(vis))
    ddims = ComradeBase.shapedims(values(dms))
    K = keys(dms)
    itp = ApplyIT{K}(Base.Fix1(ComradeBase.visibility_point, s), rotmat(axisdims(vis)))
    bvis = baseimage(vis)
    bvis .= giterate.(Ref(itp), ddims...)
    return nothing
end

function ComradeBase.centroid(im::IntensityMap{T, N}) where {T <: Reactant.RNumber, N}
    f = flux(im)
    dp = domainpoints(im)
    A = dp.transform
    dms = ComradeBase.shapedims(dp.dirs)
    itrx = ApplyIT{(:X, :Y)}(Base.Fix2(getproperty, :X), A)
    itry = ApplyIT{(:X, :Y)}(Base.Fix2(getproperty, :Y), A)

    if N == 2
        dims = Colon()
    else
        dims = (X, Y)
    end

    xcent = sum(giterate.(Ref(itrx), dms.X, dms.Y) .* im; dims = dims)
    ycent = sum(giterate.(Ref(itry), dms.X, dms.Y) .* im; dims = dims)
    return xcent ./ f, ycent ./ f
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
    res = fa.(g)
    copyto!(bvis, res)
    return nothing
end


end
