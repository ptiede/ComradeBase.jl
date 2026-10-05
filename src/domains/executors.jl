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

Return a copy of `x` whose arrays are placed on the device mesh, split along the dimensions
named in `layout` and replicated along the rest. Dim lookups stay on the host.

With a `ShardLayout`, `x` is an `IntensityMap`, a `RectiGrid`, a `StructuredDomain`, an
`AbstractDualDomain`, or a `Tuple` or `NamedTuple` of these, which is sharded element by
element. Every dimension named in `layout` must be a dimension of `x` (of either domain, for
a dual domain); for a collection, of every element.

- `IntensityMap`: its values, and the coordinates of a `StructuredDomain` it is defined on.
- `StructuredDomain`: each coordinate along the dims it spans; the executor becomes
  `ReactantEx()`.
- `RectiGrid`: returned unchanged.
- `AbstractDualDomain`: both domains, rebuilt with `Accessors.@set` on the fields
  `imgdomain` and `visdomain`. Dual domains that hold more device data add a method.

The second form passes `sharding` directly to `Reactant.to_rarray`. A dimension whose
length is not a multiple of the number of devices along its mesh axes is padded.

Requires Reactant to be loaded with its IFRT runtime (the Reactant preference
`xla_runtime = "IFRT"`) and a mesh of at least two devices.
"""
function shard end


@static if VERSION ≥ v"1.11"
    const schedulers = (:dynamic, :static, :greedy)
else
    const schedulers = (:dynamic, :static)
end

"""
    @threaded executor expr

Threads the for-loop expression `expr` using the specified `executor`, which must be `Serial()`
or a `ThreadsEx` with one of Julia's `Threads.@threads` schedulers; any other executor throws an
`ArgumentError`. When `Threads.nthreads() == 1` the loop runs as a regular for-loop.
"""
macro threaded(executor, expr)
    ex = gensym(:executor)
    threaded = nothing
    for s in schedulers
        threaded = :(
            if $ex === $(ThreadsEx){$(QuoteNode(s))}()
                Base.Threads.@threads $(QuoteNode(s)) $expr
            else
                $threaded
            end
        )
    end
    return esc(
        quote
            $ex = $(_check_threaded)($executor)
            if $ex === $(Serial)() || Base.Threads.nthreads() == 1
                $expr
            else
                $threaded
            end
        end
    )
end

macro threaded(expr)
    return esc(:($(@__MODULE__).@threaded $(ThreadsEx)() $expr))
end

_check_threaded(ex::Serial) = ex
_check_threaded(ex::ThreadsEx{S}) where {S} = S in schedulers ? ex : _throw_threaded(ex)
_check_threaded(ex) = _throw_threaded(ex)

@noinline function _throw_threaded(ex)
    throw(
        ArgumentError(
            "@threaded does not handle the executor $ex; use `Serial()` or `ThreadsEx(s)` with `s` one of $schedulers"
        )
    )
end

"""
    _pointmap!(img::IntensityMap, f, d::AbstractSingleDomain, executor)

Writes `f(domainpoints(d)[I])` into the map `img` at every index `I` of `d`, using `executor`.
Loop executors write into the storage of `img` with [`_setpoint!`](@ref). Executor extensions
add methods for their executor type; an executor without one throws an `ArgumentError`.
"""
function _pointmap!(img, f, d, ::Serial)
    dest = baseimage(img)
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

function _pointmap!(img, f, d, ::ThreadsEx{S}) where {S}
    return _threads_pointmap!(baseimage(img), f, domainpoints(d), Val(S))
end

_pointmap!(img, f, d, executor) = _throw_executor(executor)

"""
    _threads_pointmap!(dest, f, points, ::Val{S})

The loop of [`_pointmap!`](@ref) for `ThreadsEx{S}`. `S` is one of Julia's `Threads.@threads`
schedulers or `:Enzyme`, `:Polyester` when that package is loaded.
"""
_threads_pointmap!(dest, f, g, ::Val{S}) where {S} = _throw_executor(ThreadsEx(S))

@noinline function _throw_executor(executor)
    throw(
        ArgumentError(
            "the executor $executor cannot run a loop; use `Serial()`, `ThreadsEx(s)` with `s` one of $schedulers, `ThreadsEx(:Enzyme)` or `ThreadsEx(:Polyester)` with Enzyme or Polyester loaded, or an OhMyThreads scheduler with OhMyThreads loaded"
        )
    )
end

for s in schedulers
    @eval function _threads_pointmap!(dest, f, g, ::Val{$(QuoteNode(s))})
        Threads.@threads $(QuoteNode(s)) for I in _pointindices(dest, g)
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
_pointbroadcast(f::F, d::AbstractSingleDomain) where {F} = Broadcast.broadcasted(f, domainpoints(d))

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
    _broadcast_pointmap!(img::IntensityMap, f, d::AbstractSingleDomain)

The broadcasting form of [`_pointmap!`](@ref), for executors that compile array expressions
(KernelAbstractions, Reactant). A [`StokesMap`](@ref) gets one broadcast per Stokes component.
"""
function _broadcast_pointmap!(img, f::F, d) where {F}
    img .= _pointbroadcast(f, d)
    return nothing
end
