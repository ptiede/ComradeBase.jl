export Serial, ThreadsEx, ReactantEx, ShardSpec, shard_image, shard_frequency, shard_time, shardmesh,
    to_sharded

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
    ShardSpec(mesh, axes::NamedTuple)

A backend-agnostic description of how to shard image or visibility evaluation across a device
`mesh`. The `mesh` is treated opaquely by ComradeBase (at use time it is a `Reactant.Sharding.Mesh`)
so that the core package carries no Reactant dependency. `axes` maps *semantic* ComradeBase axis
names — the same names Comrade uses, `:X`, `:Y`, `:Ti`, `:Fr` — to mesh-axis names (`Symbol`s):

  - Image (`RectiGrid`/`IntensityMap`): `:X`, `:Y`, `:Ti`, `:Fr` are genuine array dimensions and map
    straight onto a sharding of those dimensions.
  - Visibility (`UnstructuredDomain`): the data is a flat vector, so `:Ti`/`:Fr` are per-point labels.
    Sharding over them means [`regroup`](@ref)ing the domain by that label so each `(Ti, Fr)` block is
    contiguous, then sharding the flat data axis. For ALMA-like data (constant baselines per stamp) the
    even split lands exactly on block boundaries; for ragged VLBI data a few blocks straddle device
    boundaries (a bounded cost — a dense pad+mask layout can remove it later).

To co-locate each `(Ti, Fr)` block's image slice with its visibilities, declare the **same** `:Ti`/`:Fr`
mesh axes on both the image and the visibility domain.

Prefer the convenience constructors on [`ReactantEx`](@ref) instead of building this directly, e.g.
`ReactantEx(mesh; X = :dx, Y = :dy)` or `ReactantEx(mesh; Fr = :dev)`.
"""
struct ShardSpec{M, A <: NamedTuple}
    mesh::M
    axes::A
end

"""
    ReactantEx()
    ReactantEx(sharding::ShardSpec)
    ReactantEx(mesh; axisname = meshaxis...)
    ReactantEx(f)                            # f(x) -> a Reactant.Sharding.AbstractSharding

Uses Reactant.jl for execution when computing the intensitymap or visibilitymap. Note that specifying
this should be unnecessary as ComradeBase will automatically switch to this backend when
it detects that it is inside a Reactant tracing context.

When a [`ShardSpec`](@ref) is attached (typically via the keyword constructor) the image or
visibility arrays are sharded across the supplied device mesh during evaluation. The keyword
constructor maps semantic axis names to mesh-axis names, e.g.

```julia
mesh = ComradeBase.shardmesh(2, 2; names = (:dx, :dy))  # requires Reactant
g = imagepixels(μas2rad(100.0), μas2rad(100.0), 256, 256; executor = ReactantEx(mesh; X = :dx, Y = :dy))
```

A mesh-axis value may itself be a tuple to shard one image/data dimension across several mesh axes,
e.g. `ReactantEx(mesh; X = (:dx1, :dx2))`.

**Escape hatch.** The semantic [`ShardSpec`](@ref) only covers the common "shard dimension *D* across
mesh axis *M*" case. For anything Reactant supports that it does not express — `NamedSharding` with
sub-axes, priorities, open/closed dims, partial replication, layouts that depend on the array — pass a
**function** instead. [`to_sharded`](@ref) calls it with the object being sharded and uses whatever
`Reactant.Sharding.AbstractSharding` it returns:

```julia
executor = ReactantEx(x -> Reactant.Sharding.NamedSharding(mesh, (:dx, nothing, :dev)))
```

See also [`shard_image`](@ref), [`shard_frequency`](@ref), and [`regroup`](@ref).
"""
struct ReactantEx{S}
    sharding::S
    # Explicit inner constructor so Julia does not auto-generate the `ReactantEx(::Any)` outer
    # constructor, which would collide with the keyword constructor below.
    ReactantEx{S}(sharding) where {S} = new{S}(sharding)
end
ReactantEx() = ReactantEx{Nothing}(nothing)
ReactantEx(s::Union{Nothing, ShardSpec, Function}) = ReactantEx{typeof(s)}(s)
function ReactantEx(mesh; axes...)
    spec = ShardSpec(mesh, NamedTuple(axes))
    return ReactantEx{typeof(spec)}(spec)
end

"""
    sharding(ex::ReactantEx)

Returns the sharding declaration attached to an executor, or `nothing` when the executor carries no
sharding (the default for [`Serial`](@ref)/[`ThreadsEx`](@ref) and for an unsharded `ReactantEx`).
"""
sharding(@nospecialize(::Any)) = nothing
sharding(ex::ReactantEx) = getfield(ex, :sharding)

"""
    shard_image(mesh; x = :dx, y = :dy)

Convenience helper returning a [`ReactantEx`](@ref) that shards the image `:X`/`:Y` axes across the
`x`/`y` axes of `mesh`.
"""
shard_image(mesh; x = :dx, y = :dy) = ReactantEx(mesh; X = x, Y = y)

"""
    shard_frequency(mesh; axis = :dev)

Convenience helper returning a [`ReactantEx`](@ref) that shards the image `:Fr` axis across the
`axis` axis of `mesh`. Combine with [`regroup`](@ref)`(visdomain, :Fr)` so each frequency block
maps to a single device.
"""
shard_frequency(mesh; axis = :dev) = ReactantEx(mesh; Fr = axis)

"""
    shard_time(mesh; axis = :dev)

Convenience helper returning a [`ReactantEx`](@ref) that shards the image `:Ti` axis across the
`axis` axis of `mesh`. Combine with [`regroup`](@ref)`(visdomain, :Ti)` so each time block maps to a
single device.
"""
shard_time(mesh; axis = :dev) = ReactantEx(mesh; Ti = axis)

"""
    shardmesh(dims...; names)

Convenience constructor for a device mesh over the available Reactant devices, laid out with the
given shape `dims` and mesh-axis `names`. Equivalent to
`Reactant.Sharding.Mesh(reshape(Reactant.devices(), dims...), names)`. Requires Reactant to be loaded.
"""
function shardmesh end

"""
    to_sharded(x)

Move a domain or map `x` onto the device mesh declared by its `ReactantEx` executor's [`ShardSpec`](@ref),
returning a sharded copy via the public `Reactant.to_rarray` API. The sharding rides on the input
arrays; ordinary evaluation code then propagates it. If the executor carries no `ShardSpec`, this is
just `Reactant.to_rarray(x)` (unsharded). Requires Reactant to be loaded.

```julia
mesh = shardmesh(length(Reactant.devices()); names = (:dev,))
dvis = domain(arrayconfig; executor = ReactantEx(mesh; Fr = :dev))
rdvis, perm = regroup(dvis, :Fr)   # sort by frequency so each Fr block is contiguous
dvis_sharded = to_sharded(rdvis)   # flat coordinate arrays now sharded across the mesh
# apply `perm` to the data/noise vectors so they stay aligned with the model visibilities
```
"""
function to_sharded end


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
