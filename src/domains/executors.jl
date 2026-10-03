export Serial, ThreadsEx, ReactantEx, ShardLayout, shard

"""
    Serial()

Uses serial execution when computing the intensitymap or visibilitymap
"""
struct Serial end

"""
    ThreadsEx(;scheduler::Symbol = :dynamic)

Uses Julia's Threads @threads macro when computing the intensitymap or visibilitymap.
You can choose from Julia's various schedulers by passing the scheduler as a parameter.
The default is :dynamic, but it isn't considered part of the stable API and may change
at any moment.
"""
struct ThreadsEx{S} end
ThreadsEx() = ThreadsEx(:dynamic)
ThreadsEx(s) = ThreadsEx{s}()

"""
    ReactantEx()

Uses Reactant.jl for execution when computing the intensitymap or visibilitymap. Specifying
this is usually unnecessary, since ComradeBase switches to this backend automatically
when it detects that it is inside a Reactant tracing context.
"""
struct ReactantEx end

"""
    ShardLayout(mesh; axes...)

Describes how to place an object on a device `mesh`. Each keyword maps a dimension name of
the object (`X`, `Y`, `Ti`, `Fr`, ...) to the name of a mesh axis, or to a tuple of mesh-axis
names, along which that dimension is split. Dimensions that are not named are replicated.
The `mesh` is not inspected here; [`shard`](@ref) requires it to be a
`Reactant.Sharding.Mesh`.

```julia
layout = ShardLayout(mesh; Ti = :t, Fr = :f)
```
"""
struct ShardLayout{M, A <: NamedTuple}
    mesh::M
    axes::A
    function ShardLayout{M, A}(mesh, axes) where {M, A <: NamedTuple}
        isempty(axes) && throw(ArgumentError("ShardLayout requires at least one dimension => mesh-axis assignment, e.g. `ShardLayout(mesh; X = :x)`"))
        for (k, v) in pairs(axes)
            _ismeshaxes(v) || throw(ArgumentError("ShardLayout: the value for dimension `$k` must be a `Symbol` or a tuple of `Symbol`s naming mesh axes, got $(repr(v))"))
        end
        return new{M, A}(mesh, axes)
    end
end
ShardLayout(mesh, axes::NamedTuple) = ShardLayout{typeof(mesh), typeof(axes)}(mesh, axes)
ShardLayout(mesh; axes...) = ShardLayout(mesh, NamedTuple(axes))

_ismeshaxes(::Symbol) = true
_ismeshaxes(v::Tuple) = !isempty(v) && all(x -> x isa Symbol, v)
_ismeshaxes(_) = false

"""
    shard(x, layout::ShardLayout)
    shard(x, sharding::Reactant.Sharding.AbstractSharding)

Return a copy of `x` whose values are placed on the device mesh, split along the dimensions
named in `layout`. For an `IntensityMap` only the pixel values are placed; the grid stays on
the host. The second form passes `sharding` directly to `Reactant.to_rarray`. A dimension
whose length is not a multiple of the number of devices along its mesh axes is padded.

Requires Reactant to be loaded with its IFRT runtime (the Reactant preference
`xla_runtime = "IFRT"`) and a mesh of at least two devices.
"""
function shard end


#TODO can this be made nicer?
@static if VERSION ≥ v"1.11"
    const schedulers = (:(:dynamic), :(:static), :(:greedy))
else
    const schedulers = (:(:dynamic), :(:static))
end

"""
    @threaded executor expr

Threads the for-loop expression `expr` using the specified `executor`. The executor must be one of
`ThreadsEx` or `Serial`. Note that if the `Threads.nthreads() == 1` we automatically default to 
a regular for-loop to prevent overhead.
"""
macro threaded(executor, expr)
    return esc(
        quote
            if Threads.nthreads() > 1 && $(executor) != Serial()
                if $(executor) == ThreadsEx{:static}()
                    Threads.@threads :static $(expr)
                elseif $(executor) == ThreadsEx{:dynamic}()
                    Threads.@threads :dynamic $(expr)
                end
            else
                $(expr)
            end
        end
    )
end

macro threaded(expr)
    return :(@threaded(ThreadsEx(), $(expr)))
end

"""
    _pointmap!(dest, f, d::AbstractSingleDomain, executor)

Writes `f(domainpoints(d)[I])` into the storage `dest` at every index `I` of `d` with
[`_setpoint!`](@ref), using `executor`. Executor extensions add methods for their executor type.
"""
function _pointmap!(dest, f, d, ::Serial)
    g = domainpoints(d)
    for I in _pointindices(dest, g)
        _setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

"""
    _pointindices(dest, g)

Returns `CartesianIndices(g)` after checking that the leading axes of the storage `dest` are
the axes of the points `g`.
"""
function _pointindices(dest, g)
    lead = ntuple(k -> axes(dest, k), Val(ndims(g)))
    lead == axes(g) || throw(
        DimensionMismatch("map storage with axes $(axes(dest)) does not start with the axes $(axes(g)) of the domain")
    )
    return CartesianIndices(g)
end

"""
    _setpoint!(dest, I::CartesianIndex, v)

Writes the point value `v` into the storage `dest` at the domain index `I`. A number is stored
at `dest[I]`. A `StaticArray` such as `StokesParams` is stored at `dest[I, k]` for every index
`k` of `v`; the dims of `dest` after those of `I` must have the size of `v`.
"""
@inline function _setpoint!(dest::AbstractArray{<:Any, M}, I::CartesianIndex{M}, v::Number) where {M}
    dest[I] = v
    return dest
end

@inline function _setpoint!(dest, I::CartesianIndex{M}, v::StaticArray) where {M}
    _trailingsize(dest, Val(M)) == size(v) || _throw_pointsize(dest, Val(M), v)
    ks = CartesianIndices(v)
    ntuple(n -> (dest[I, ks[n]] = v[n]), Val(length(v)))
    return dest
end

_setpoint!(dest, I::CartesianIndex{M}, v) where {M} = _throw_pointsize(dest, Val(M), v)

_trailingsize(dest, ::Val{M}) where {M} = ntuple(k -> size(dest, M + k), Val(ndims(dest) - M))

@noinline function _throw_pointsize(dest, ::Val{M}, v) where {M}
    throw(
        DimensionMismatch(
            "a point value of type $(typeof(v)) cannot fill the trailing dims of size $(_trailingsize(dest, Val(M))) of the map storage"
        )
    )
end

function _pointmap!(dest, f, d, ::ThreadsEx{S}) where {S}
    return _threads_pointmap!(dest, f, domainpoints(d), Val(S))
end

"""
    _threads_pointmap!(dest, f, points, ::Val{S})

The loop of [`_pointmap!`](@ref) for `ThreadsEx{S}`. `S` is one of Julia's `Threads.@threads`
schedulers or `:Enzyme`, `:Polyester` when that package is loaded.
"""
function _threads_pointmap! end

for s in schedulers
    @eval function _threads_pointmap!(dest, f, g, ::Val{$s})
        Threads.@threads $s for I in _pointindices(dest, g)
            _setpoint!(dest, I, f(g[I]))
        end
        return nothing
    end
end

"""
    NamedPointFn{K}(f)

Holds a point function `f`; `_applynamed(p, xs...)` applies `f` to `NamedTuple{K}(xs)` for the
positional values `xs`, so that `f` can be broadcast over coordinate arrays.
"""
struct NamedPointFn{K, F}
    f::F
end
NamedPointFn{K}(f) where {K} = NamedPointFn{K, typeof(f)}(f)
@inline _applynamed(p::NamedPointFn{K}, xs...) where {K} = p.f(NamedTuple{K}(xs))

"""
    _pointbroadcast(f, d::AbstractSingleDomain)

Returns the lazy broadcast of `f` over the points of `d`, with the axes of `d`.
"""
_pointbroadcast(f, d::AbstractSingleDomain) = Broadcast.broadcasted(f, domainpoints(d))

"""
    _broadcast_pointmap!(dest, f, d::AbstractSingleDomain)

The broadcasting form of [`_pointmap!`](@ref), for executors that compile array expressions
(KernelAbstractions, Reactant): one broadcast per entry of the trailing dims of `dest`.
"""
function _broadcast_pointmap!(dest, f, d)
    _foreach_component(dest, f, Val(ndims(d))) do slab, fk
        Broadcast.materialize!(slab, _pointbroadcast(fk, d))
    end
    return nothing
end

"""
    ComponentFn(f, k)

Returns entry `k` of the point value of `f`, in the order of [`_setpoint!`](@ref).
"""
struct ComponentFn{F}
    f::F
    k::Int
end
(c::ComponentFn)(xs...) = c.f(xs...)[c.k]

"""
    _foreach_component(h, dest, f, ::Val{M})

Calls `h(slab, fk)` for every entry of the dims of `dest` after the first `M`, where `slab` is
the view of `dest` at that entry and `fk` the matching [`ComponentFn`](@ref) of `f`. Without
trailing dims it calls `h(dest, f)`.
"""
function _foreach_component(h, dest, f, ::Val{M}) where {M}
    if ndims(dest) == M
        h(dest, f)
        return nothing
    end
    trailing = CartesianIndices(ntuple(k -> axes(dest, M + k), Val(ndims(dest) - M)))
    for (n, k) in enumerate(trailing)
        h(_slab(dest, Tuple(k)...), ComponentFn(f, n))
    end
    return nothing
end
