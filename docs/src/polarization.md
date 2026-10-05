# Polarized maps

```@meta
CurrentModule = ComradeBase
```

Polarization is stored in trailing dims of a map's storage, after the dims of its domain.
The element type of a polarized map is a plain number.

## Stokes maps

A [`StokesMap`](@ref) has a trailing [`Stokes`](@ref) dim of length 4 holding Stokes I, Q,
U and V in that order. `IntensityMap(storage, domain, Stokes())` wraps storage of size
`(size(domain)..., 4)` without copying:

```@example pol
using ComradeBase

g = RectiGrid((X(range(-10.0, 10.0; length = 8)), Y(range(-10.0, 10.0; length = 8))))
P = rand(8, 8, 4)
img = IntensityMap(P, g, Stokes())
img isa StokesMap, size(img), axisdims(img) === g
```

[`stokes`](@ref) returns one component as an unpolarized map over the same domain, viewing
the same storage. DimensionalData selection along `Stokes` does the same:

```@example pol
stokes(img, :Q) == img[Stokes(2)]
```

[`baseimage`](@ref) returns the dense storage. [`eldims`](@ref) returns the trailing dims.

```@example pol
baseimage(img) === P, eldims(img)
```

Base reductions and element-wise functions act on the numbers across all dims, Stokes
included: `sum(img)` adds all four components together. [`flux`](@ref) sums each component
separately and returns a `StokesParams`; [`centroid`](@ref) and [`second_moment`](@ref) use
Stokes I.

```@example pol
flux(img)
```

An array of `StokesParams`, including a `StructArray` of them, is copied into dense storage:

```@example pol
sp = [StokesParams(1.0, 0.1, 0.2, 0.05) for _ in 1:8, _ in 1:8]
IntensityMap(sp, g) isa StokesMap
```

A polarized model's point functions return a `StokesParams`; the executors write its four
fields into the trailing `Stokes` entries of the storage.

## Coherency maps

Data in the local feed bases of the two antennas of a baseline are 2×2 coherency matrices.
A [`CoherencyMap`](@ref) stores them in two trailing feed dims [`Fa`](@ref) and
[`Fb`](@ref), each of length 2: the entry `[..., a, b]` is the element `e_ab`, with `a`
the feed of antenna a and `b` the feed of antenna b.

The feed dims carry no polarization basis. [`coherencymap`](@ref) converts Stokes
parameters to coherencies in a basis given as an argument (`CirBasis()` or `LinBasis()`),
and [`stokesmap`](@ref) converts back:

```@example pol
c = coherencymap(img, CirBasis())
c isa CoherencyMap, size(c)
```

```@example pol
stokesmap(c, CirBasis()) ≈ img
```

[`coherency`](@ref) selects one element as an unpolarized map:

```@example pol
coherency(c, 1, 2) == c[Fa(1), Fb(2)]
```

The in-place forms [`coherencymap!`](@ref) and [`stokesmap!`](@ref) need complex storage,
overwrite it and return a map over a reshape of it; the argument's storage is consumed.
