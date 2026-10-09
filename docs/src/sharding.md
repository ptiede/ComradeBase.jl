# Sharding with Reactant

```@meta
CurrentModule = ComradeBase
```

With [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) loaded, [`shard`](@ref) places
maps and domains on a mesh of devices, split along named dims. Code compiled with
`Reactant.@jit` then runs as one SPMD program across the mesh. The examples on this page
are not run when the docs are built, because they need several devices.

## Requirements

- Reactant's IFRT runtime. Set the preference in the `LocalPreferences.toml` of the active
  environment:

  ```toml
  [Reactant]
  xla_runtime = "IFRT"
  ```

  The PJRT runtime cannot return wrapped arrays from a sharded `@jit` call.
- A mesh of at least two devices; `shard` throws on a single-device mesh. On a machine
  without several accelerators, the CPU can be split into simulated devices for testing by
  setting `XLA_FLAGS=--xla_force_host_platform_device_count=4` before Julia starts and
  calling `Reactant.set_default_backend("cpu")`.

## Layouts

A [`ShardLayout`](@ref) maps dim names of the object to mesh axes. Dims that are not
named are replicated. Every dim named in a layout must be a dim of the object `shard` is
called on; a name that matches nothing is an error, not a no-op.

```julia
using ComradeBase, Reactant

mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices(), 2, 2), (:t, :f))
layout = ShardLayout(mesh; Ti = :t, Fr = :f)
```

What `shard` places for each object:

- `IntensityMap`: its storage. The trailing Stokes or feed dims of the storage of a
  polarized map are always replicated. A map over a `StructuredDomain` also has its domain
  coordinates placed.
- `StructuredDomain`: each coordinate array, split along the layout dims it spans and
  replicated along the rest. The executor becomes `ReactantEx()`.
- `RectiGrid`: returned unchanged. Dim lookups always stay on the host.
- `AbstractDualDomain`: both domains; the layout may name a dim of either one.
- `Tuple` or `NamedTuple`: each element, checked against the layout on its own.

A dim whose length is not a multiple of the number of devices along its mesh axes is
padded by the IFRT runtime.

## Movies across devices

A multi-frequency movie is an image stack `(X, Y, Ti, Fr)`. It is sharded along `Ti` and
`Fr`. The data stay in their natural layout: one `(Pt, Fr)` domain whose points carry an
observation time in a `Ti` coordinate. The points are replicated along `Pt`, and the
domain's `Fr` dim is sharded on the same mesh axis as the image's `Fr` dim:

```julia
npt = 200
frs = Fr([86.0e9, 230.0e9])
d = StructuredDomain(
    (Pt(npt), frs);
    u = 4.0e6 .* randn(npt), v = 4.0e6 .* randn(npt), Ti = 4 .* rand(npt),
)
sd = shard(d, ShardLayout(mesh; Fr = :f))

nx = 16
fov = 1.0e-9
g = spatialgrid(fov, fov, nx, nx) ⊗ Ti([0.0, 1.0, 2.0, 3.0]) ⊗ frs
img = IntensityMap(rand(nx, nx, 4, 2), g)
simg = shard(img, layout)
```

The domain gets its own layout because it has no `Ti` dim. An `AbstractDualDomain` that
pairs `g` with `d` accepts the combined layout, since it has the dims of both.

An analytic model `m` is evaluated on the sharded domain directly; the result is split along
`Fr`:

```julia
vis = @jit visibilitymap(m, sd)
```

The step from image planes to visibilities needs each point's frame. Assigning frames from
the `Ti` coordinate and the image's `Ti` lookup is up to the caller; here a point belongs to
the last frame that starts at or before its time:

```julia
frame = searchsortedlast.(Ref(collect(g.Ti)), d.Ti)
mask = frame .== (1:4)'
```

Each device then transforms only the frames it holds, for every point, and masks out the
points that belong to other frames. Summing over `Ti` becomes a single all-reduce of the
per-point values; the stack itself is never gathered onto one device. This example uses a
direct Fourier transform written with broadcasting:

```julia
xs = vec(collect(g.X) .+ 0 .* collect(g.Y)')
ys = vec(0 .* collect(g.X) .+ collect(g.Y)')
A = cis.(-2π .* (reshape(d.U, npt, 1, 2) .* xs' .+ reshape(d.V, npt, 1, 2) .* ys'))

function movievis(P, A, mask)
    nx, ny, nt, nf = size(P)
    planes = reshape(P, 1, nx * ny, nt, nf)
    vt = sum(reshape(A, size(A, 1), nx * ny, 1, nf) .* planes; dims = 2)
    return dropdims(sum(reshape(mask, size(mask, 1), 1, nt) .* vt; dims = (2, 3)); dims = (2, 3))
end

Ar = Reactant.to_rarray(A; sharding = Reactant.Sharding.DimsSharding(mesh, (3,), (:f,)))
maskr = Reactant.to_rarray(mask; sharding = Reactant.Sharding.Replicated(mesh))
v = @jit movievis(baseimage(simg), Ar, maskr)
```

A gather from a gridded Fourier transform at each point's frame index partitions the same
way: a local gather on each device, then one all-reduce. Every device still evaluates every
point, so the per-device cost grows with the number of points.

## Gradients

Pass a gradient shadow that is already sharded like the primal. A shadow created inside
the compiled function, for example with `zero`, comes back replicated
([EnzymeAD/Reactant.jl#3398](https://github.com/EnzymeAD/Reactant.jl/issues/3398)).

```julia
using Enzyme

loss(P, A, mask) = sum(abs2, movievis(P, A, mask))
function grad!(dP, P, A, mask)
    Enzyme.autodiff(Reverse, loss, Active, Duplicated(P, dP), Const(A), Const(mask))
    return dP
end

dP = baseimage(shard(zero(img), layout))
@jit grad!(dP, baseimage(simg), Ar, maskr)
```

The gradient stays split along `Ti` and `Fr`.

## Several datasets

Several datasets, for example two arrays observing the same movie, are a `Tuple` or
`NamedTuple` of domains, each with its own image grid. `shard` places each element and
checks the layout against each one:

```julia
sds = shard((; eht = d1, ngeht = d2), ShardLayout(mesh; Fr = :f))
```

## Known limitations

- Keep a whole computation inside one `@jit` call. Passing a sharded map over a
  `StructuredDomain` returned by one `@jit` call into a second one currently fails with
  Reactant's `TODO(#2234)` error, because the returned coordinates lose their replicated
  sharding ([EnzymeAD/Reactant.jl#3399](https://github.com/EnzymeAD/Reactant.jl/issues/3399)).
- XLA's partitioner replicates FFTs, even when only batch dims are sharded.
- A gather of a window of pixels around each point all-reduces the whole window instead of
  the interpolated value
  ([EnzymeAD/Reactant.jl#3396](https://github.com/EnzymeAD/Reactant.jl/issues/3396)).
