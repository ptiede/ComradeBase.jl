module ComradeBaseReactantExt

using ComradeBase
using Reactant
using StaticArrays

import ComradeBase: AbstractSingleDomain, basedim, dims
using ComradeBase: AbstractDualDomain, AbstractRectiGrid, DD, ReactantEx, ShardLayout, StructuredDomain
using Accessors: @set
import Reactant: AnyTracedRArray, TracedRArray, unwrapped_eltype

const Shardable = Union{IntensityMap, AbstractRectiGrid, StructuredDomain, AbstractDualDomain}

function ComradeBase.shard(x::Shardable, layout::ShardLayout)
    _check_layout(x, layout)
    return _place(x, layout)
end

ComradeBase.shard(xs::Union{Tuple, NamedTuple}, layout::ShardLayout) = map(x -> shard(x, layout), xs)

function ComradeBase.shard(x, sh::Reactant.Sharding.AbstractSharding)
    _check_sharding_supported(sh)
    return Reactant.to_rarray(x; sharding = sh)
end
function ComradeBase.shard(img::IntensityMap, sh::Reactant.Sharding.AbstractSharding)
    _check_sharding_supported(sh)
    return ComradeBase._rewrap(img, Reactant.to_rarray(baseimage(img); sharding = sh))
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

_dimnames(x::Union{IntensityMap, AbstractRectiGrid, StructuredDomain}) = map(DD.name, dims(x))
_dimnames(d::AbstractDualDomain) = (_dimnames(imgdomain(d))..., _dimnames(visdomain(d))...)

function _check_layout(x, layout::ShardLayout)
    _check_runtime()
    mesh = layout.mesh
    _check_mesh(mesh)
    dnames = _dimnames(x)
    for (dname, v) in pairs(layout.axes)
        dname in dnames || throw(ArgumentError("ShardLayout dimension `$dname` is not a dimension of the $(nameof(typeof(x))); available dimensions are $(Tuple(unique(dnames)))"))
        for m in _meshaxes(v)
            m in mesh || throw(ArgumentError("ShardLayout mesh axis `$m` (for dimension `$dname`) is not in the mesh; available mesh axes are $(mesh.axis_names)"))
        end
    end
    return nothing
end

# The sharding of an array whose dims are named `dnames`; layout dims it lacks are ignored.
function _sharding(dnames, layout::ShardLayout)
    ks = filter(in(dnames), keys(layout.axes))
    isempty(ks) && return Reactant.Sharding.Replicated(layout.mesh)
    positions = map(k -> findfirst(==(k), dnames), ks)
    return Reactant.Sharding.DimsSharding(layout.mesh, positions, map(k -> layout.axes[k], ks))
end

function _place(img::IntensityMap, layout::ShardLayout)
    vals = Reactant.to_rarray(baseimage(img); sharding = _sharding(_dimnames(img), layout))
    return IntensityMap(vals, _place(axisdims(img), layout), DD.refdims(img), DD.name(img))
end

_place(g::AbstractRectiGrid, ::ShardLayout) = g

function _place(d::StructuredDomain, layout::ShardLayout)
    cs = map(ComradeBase.coords(d), ComradeBase.coordspans(d)) do c, span
        Reactant.to_rarray(c; sharding = _sharding(span, layout))
    end
    return ComradeBase.rebuild(d; coords = cs, executor = ReactantEx())
end

function _place(d::AbstractDualDomain, layout::ShardLayout)
    d = @set d.imgdomain = _place(imgdomain(d), layout)
    return @set d.visdomain = _place(visdomain(d), layout)
end

# Tracing paths into an `IntensityMap` address struct fields, not array elements.
Reactant.traced_getfield(@nospecialize(obj::IntensityMap), field) = getfield(obj, field)

# The element type and dims type of a traced map follow its traced data and domain.
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(M::Type{<:IntensityMap}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    M isa DataType || return M
    T, N, D, G, A, R, Na = M.parameters
    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, ndevices, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, ndevices, runtime)
    R2 = Reactant.traced_type_inner(R, seen, mode, track_numbers, ndevices, runtime)
    return IntensityMap{eltype(A2), N, _dimstype(G2), G2, A2, R2, Na}
end

_dimstype(::Type{<:AbstractSingleDomain{D}}) where {D} = D

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

# A range at a traced index steps from its start instead of calling `getindex`: Reactant's
# own range indexing lowers a `LinRange` lookup to `Base.lerpi`, which accepts only plain
# integers.
Base.@propagate_inbounds function ComradeBase.rgetindex(I::AbstractRange, i::TInt)
    return first(I) + (i - one(i)) * step(I)
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


function Base.eltype(d::ComradeBase.AbstractRectiGrid{D, E}) where {D, E <: ReactantEx}
    return Reactant.allowscalar() do
        eltype(basedim(first(dims(d))))
    end
end

function ComradeBase._storage(::ReactantEx, ::Type{T}, sz) where {T}
    return similar(TracedRArray{unwrapped_eltype(T)}, sz)
end

# A StructuredDomain traces its coordinates and executor; its dims, spans and header stay on
# the host and are compile-time constants of a traced function.
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:StructuredDomain}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    D, C, S, E, H = T.parameters
    C2 = Reactant.traced_type_inner(C, seen, mode, track_numbers, ndevices, runtime)
    E2 = Reactant.traced_type_inner(E, seen, mode, track_numbers, ndevices, runtime)
    return StructuredDomain{D, C2, S, E2, H}
end

Base.@nospecializeinfer function Reactant.make_tracer(
        seen,
        @nospecialize(prev::StructuredDomain),
        @nospecialize(path),
        mode;
        @nospecialize(sharding = Reactant.Sharding.NoSharding()),
        kwargs...
    )
    cs = ComradeBase.coords(prev)
    ex = ComradeBase.executor(prev)
    ci = Base.fieldindex(StructuredDomain, :coords)
    ei = Base.fieldindex(StructuredDomain, :executor)
    if mode == Reactant.TracedToTypes
        push!(path, Core.Typeof(prev))
        push!(path, map(d -> collect(basedim(d)), dims(prev)))
        push!(path, ComradeBase.coordspans(prev))
        push!(path, ComradeBase.header(prev))
        Reactant.make_tracer(seen, cs, path, mode; sharding = Base.getproperty(sharding, ci), kwargs...)
        Reactant.make_tracer(seen, ex, path, mode; sharding = Base.getproperty(sharding, ei), kwargs...)
        return nothing
    end
    tcs = Reactant.make_tracer(
        seen, cs, Reactant.append_path(path, ci), mode;
        sharding = Base.getproperty(sharding, ci), kwargs...
    )
    tex = Reactant.make_tracer(
        seen, ex, Reactant.append_path(path, ei), mode;
        sharding = Base.getproperty(sharding, ei), kwargs...
    )
    return StructuredDomain(dims(prev), tcs, ComradeBase.coordspans(prev), tex, ComradeBase.header(prev))
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


# `f` is passed in a `Ref` broadcast argument: broadcasting a callable that holds traced
# values (a model with traced parameters) fails with
# `AssertionError: input shapes are Tuple{Vararg{Int64}}[(), (8, 6), (8, 6)]`.
function ComradeBase.pointbroadcasted(f::F, d::RectiGrid{<:Any, ReactantEx}) where {F}
    dms = map(Reactant.materialize_traced_array ∘ basedim, named_dims(d))
    itp = ApplyIT{keys(dms)}(f, rotmat(d))
    return Broadcast.broadcasted(giterate, Ref(itp), ComradeBase.shapedims(values(dms))...)
end

function ComradeBase.centroid(img::ComradeBase.RectiMap{T}) where {T <: Reactant.RNumber}
    f = flux(img)
    g = axisdims(img)
    dims = ndims(img) == 2 ? Colon() : (X, Y)
    xcent = sum(ComradeBase.pointbroadcasted(Base.Fix2(getproperty, :X), g) .* img; dims)
    ycent = sum(ComradeBase.pointbroadcasted(Base.Fix2(getproperty, :Y), g) .* img; dims)
    return xcent ./ f, ycent ./ f
end


function ComradeBase._pointmap!(img, f::F, d, ::ReactantEx) where {F}
    return ComradeBase._broadcast_pointmap!(img, f, d)
end

end
