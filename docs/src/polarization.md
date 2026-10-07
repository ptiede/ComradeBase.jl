# Polarized maps

```@meta
CurrentModule = ComradeBase
```

The element type of a map carries its polarization. A [`StokesMap`](@ref) is an
`IntensityMap` whose elements are `StokesParams`; a [`CoherencyMap`](@ref) is one whose
elements are 2×2 `SMatrix` coherency matrices. The dims of a polarized map are the dims of
its domain.

## Stokes maps

The data of a Stokes map can be any array of `StokesParams`. ComradeBase allocates a
`FieldDimArray` over dense storage of size `(size(domain)..., 4)`, holding Stokes I, Q, U
and V in that order along the last dim. Existing dense storage is wrapped without copying:

```@example pol
using ComradeBase

g = RectiGrid((X(range(-10.0, 10.0; length = 8)), Y(range(-10.0, 10.0; length = 8))))
P = rand(8, 8, 4)
img = IntensityMap(FieldDimArray{StokesParams}(P), g)
img isa StokesMap, size(img), axisdims(img) === g
```

Indexing returns a `StokesParams`, and DimensionalData selection works along the domain
dims:

```@example pol
img[X = 1, Y = 2] == img[1, 2]
```

[`stokes`](@ref) returns one component as an unpolarized map over the same domain. For
`FieldDimArray` and `StructArray` data it is a view of the data:

```@example pol
q = stokes(img, :Q)
q == P[:, :, 2], parent(baseimage(q)) === P
```

A `StructArray` of `StokesParams` with a concrete element type also gives a Stokes map:

```@example pol
using StructArrays
sp = StructArray{StokesParams{Float64}}((rand(8, 8), rand(8, 8), rand(8, 8), rand(8, 8)))
IntensityMap(sp, g) isa StokesMap
```

[`flux`](@ref) sums each Stokes component separately and returns a `StokesParams`, or a
map of them when the map has dims besides `X` and `Y`. [`centroid`](@ref) and
[`second_moment`](@ref) use Stokes I.

```@example pol
flux(img)
```

Base reductions such as `sum` act on the elements. They work on the CPU; on a GPU or under
Reactant, use [`stokes`](@ref) to reduce each component.

A polarized model's point functions return a `StokesParams`, and the executors write it
into the element of the map at each point.

## Coherency maps

Data in the local feed bases of the two antennas of a baseline are 2×2 coherency matrices.
The element `[a, b]` of a coherency matrix is `e_ab`, with `a` the feed of antenna a and
`b` the feed of antenna b. The matrices carry no polarization basis.

[`coherencymap`](@ref) converts Stokes parameters to coherencies in a basis given as an
argument (`CirBasis()` or `LinBasis()`), and [`stokesmap`](@ref) converts back. Both are
element-wise broadcasts, so the data of the result follow from the data of the argument:
a `FieldDimArray` gives a `FieldDimArray` over storage of size `(size(domain)..., 2, 2)`.

```@example pol
c = coherencymap(img, CirBasis())
c isa CoherencyMap, size(c), size(parent(baseimage(c)))
```

```@example pol
stokesmap(c, CirBasis()) ≈ img
```

[`coherency`](@ref) returns one element as an unpolarized map:

```@example pol
coherency(c, 1, 2) == getindex.(c, 1, 2)
```

## Sharding

[`shard`](@ref) places the dense storage of a polarized map. Its trailing Stokes or feed
dims are not dims of the map and are always replicated; a [`ShardLayout`](@ref) names only
domain dims.
