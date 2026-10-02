module ComradeBaseReactantExt

using ComradeBase
using StructArrays
using Reactant
using StaticArrays

import ComradeBase: AbstractSingleDomain, basedim, dims, UnstructuredMap
using ComradeBase: ReactantEx, ShardLayout
import Reactant: AnyTracedRArray, TracedRArray, unwrapped_eltype

function ComradeBase.shard(img::IntensityMap, layout::ShardLayout)
    vals = Reactant.to_rarray(baseimage(img); sharding = _dimssharding(img, layout))
    return IntensityMap(vals, ComradeBase.axisdims(img))
end

function ComradeBase.shard(x, sh::Reactant.Sharding.AbstractSharding)
    _check_sharding_supported(sh)
    return Reactant.to_rarray(x; sharding = sh)
end
function ComradeBase.shard(img::IntensityMap, sh::Reactant.Sharding.AbstractSharding)
    _check_sharding_supported(sh)
    return IntensityMap(Reactant.to_rarray(baseimage(img); sharding = sh), ComradeBase.axisdims(img))
end

function _check_runtime()
    runtime = Reactant.XLA.REACTANT_XLA_RUNTIME
    runtime == "IFRT" && return nothing
    throw(
        ArgumentError(
            "sharding requires Reactant's IFRT runtime, but the active runtime is $runtime " *
                "(see EnzymeAD/Reactant.jl#2989). Set the Reactant preference `xla_runtime = \"IFRT\"`, " *
                "e.g. in LocalPreferences.toml, and restart Julia."
        )
    )
end

function _check_mesh(mesh)
    mesh isa Reactant.Sharding.Mesh ||
        throw(ArgumentError("ShardLayout mesh must be a `Reactant.Sharding.Mesh`, got $(typeof(mesh))"))
    length(mesh) > 1 ||
        throw(ArgumentError("single-device meshes are not supported by Reactant; use a mesh with at least 2 devices or do not shard"))
    return nothing
end

function _check_sharding_supported(sh)
    _check_runtime()
    s = Reactant.Sharding.unwrap_shardinfo(sh)
    hasfield(typeof(s), :mesh) && _check_mesh(s.mesh)
    return nothing
end

_meshaxes(v::Symbol) = (v,)
_meshaxes(v::Tuple) = v

function _dimssharding(img::IntensityMap, layout::ShardLayout)
    _check_runtime()
    mesh = layout.mesh
    _check_mesh(mesh)
    dnames = keys(ComradeBase.named_dims(img))
    positions = map(keys(layout.axes), values(layout.axes)) do dname, v
        p = findfirst(==(dname), dnames)
        p === nothing && throw(ArgumentError("ShardLayout dimension `$dname` is not a dimension of the image; available dimensions are $dnames"))
        for m in _meshaxes(v)
            m in mesh || throw(ArgumentError("ShardLayout mesh axis `$m` (for dimension `$dname`) is not in the mesh; available mesh axes are $(mesh.axis_names)"))
        end
        return p
    end
    return Reactant.Sharding.DimsSharding(mesh, positions, values(layout.axes))
end

# Tracing paths into an `IntensityMap` address struct fields, not array elements.
Reactant.traced_getfield(@nospecialize(obj::IntensityMap), field) = getfield(obj, field)

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
