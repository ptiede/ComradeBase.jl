# Domains and maps

```@meta
CurrentModule = ComradeBase
```

A domain says where a model is evaluated. ComradeBase has two kinds:

- [`RectiGrid`](@ref): a rectilinear image grid with spatial dims `X`, `Y` and optional
  `Ti` (time) and `Fr` (frequency) dims.
- [`StructuredDomain`](@ref): a set of points, such as the baselines of a VLBI observation,
  with an index dim `Pt` and optional plane dims.

[`intensitymap`](@ref) and [`visibilitymap`](@ref) evaluate a model on a domain and
return an [`IntensityMap`](@ref), a `DimensionalData.AbstractDimArray` whose dims are those
of the domain.

## A model to evaluate

The examples on this page use a circular Gaussian with analytic intensity and visibility
functions. A model defines `intensity_point` and `visibility_point`, which take one point
of a domain as a `NamedTuple` of coordinates.

```@example domains
using ComradeBase

struct Gaussian{T} <: ComradeBase.AbstractModel
    σ::T
end

ComradeBase.visanalytic(::Type{<:Gaussian}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:Gaussian}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:Gaussian}) = ComradeBase.NotPolarized()

function ComradeBase.intensity_point(m::Gaussian, p)
    (; X, Y) = p
    return exp(-(X^2 + Y^2) / (2 * m.σ^2)) / (2π * m.σ^2)
end

function ComradeBase.visibility_point(m::Gaussian, p)
    (; U, V) = p
    return complex(exp(-2π^2 * m.σ^2 * (U^2 + V^2)))
end
nothing # hide
```

## Image grids

[`spatialgrid`](@ref) builds the `(X, Y)` grid of pixel centers from a field of view and
a pixel count:

```@example domains
g = spatialgrid(20.0, 20.0, 64, 64)
img = intensitymap(Gaussian(2.0), g)
size(img)
```

Non-spatial dims are appended with [`gridproduct`](@ref), or its alias `⊗`; the map then has one
image plane per time and frequency. The order of the factors is the memory layout:

```@example domains
gm = spatialgrid(20.0, 20.0, 32, 32) ⊗ Ti([0.0, 0.5, 1.0]) ⊗ Fr([230.0e9, 345.0e9])
size(intensitymap(Gaussian(2.0), gm))
```

## Point domains

A `StructuredDomain` is built from a tuple of dims and keyword coordinate arrays. The first
dim is always `Pt`, the point index; `Pt(n)` is shorthand for `Pt(Base.OneTo(n))`. A
domain with only `Pt` is a list of points:

```@example domains
U = 0.1 .* randn(100)
V = 0.1 .* randn(100)
d = StructuredDomain((Pt(100),); U, V)
vis = visibilitymap(Gaussian(2.0), d)
size(vis)
```

`UnstructuredDomain((; U, V))` and `StructuredDomain((; U, V))` build the same `(Pt,)`
domain from a `NamedTuple`.

Plane dims such as `Fr` follow `Pt`. Each coordinate array spans a subset of the dims,
in the order they appear in the domain, and its axes must match those dims. The span is
inferred from the array's size when only one ordered subset of the dims matches it;
otherwise it is given as `array => (:Pt, :Fr)`.

Here each row has a measurement at every frequency. Baselines are given in meters as `u`,
`v`, so they vary with `Pt` only and are converted to wavelengths with the `Fr` lookup
(in Hz). A per-point observation time `Ti` spans `Pt`:

```@example domains
npt = 50
dfr = StructuredDomain(
    (Pt(npt), Fr([230.0e9, 345.0e9]));
    u = 4.0e6 .* randn(npt), v = 4.0e6 .* randn(npt), Ti = sort(rand(npt)),
)
ComradeBase.coordspans(dfr)
```

```@example domains
size(dfr.U)
```

The coordinates `U`, `V` (wavelengths), `u`, `v` (meters), `Ti`, `Fr` and `valid` (a `Bool`
mask) have a fixed meaning. Any other coordinate, for example antenna indices or feed
rotation angles, is passed through to the point `NamedTuple` unchanged.

```@example domains
first(domainpoints(dfr))
```

A map over a `StructuredDomain` is indexed by dim name. Slicing along `Pt` or `Fr` slices the
coordinates with the values:

```@example domains
visfr = visibilitymap(Gaussian(2.0), dfr)
sub = visfr[Pt(1:10), Fr(2)]
size(sub), size(axisdims(sub).U)
```

### Movies and frame assignment

Points are not grouped by frame. A movie's points stay in one `(Pt,)` or `(Pt, Fr)`
domain, and the `Ti` coordinate records each point's observation time. The image side
of a movie is a grid with a `Ti` dim; the step that computes visibilities from the image
planes looks up each point's frame from its `Ti` coordinate. A `(Pt, Ti, Fr)` domain can
be constructed, but nothing requires one.

[`frameindex`](@ref) does that lookup. For a dim with `Intervals` sampling, such as one
built by [`frames`](@ref), a point belongs to the interval that contains it; otherwise
its coordinate must equal a plane's value. A point that matches no plane is an error.

```@example domains
scans = frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])
gmovie = spatialgrid(20.0, 20.0, 32, 32) ⊗ scans
frameindex(scans, [0.2, 2.0, 4.9])
```

### Several datasets

Several arrays, bands or sources are a `Tuple` or `NamedTuple` of domains. Each is dense
on its own and has its own image grid; the datasets of one movie share the `Ti` dim of the
grid. [`shard`](@ref) works element by element on such a collection.

## Executors

A domain's executor chooses how a map is computed: `Serial()` (the default), `ThreadsEx()`
for Julia threads, a KernelAbstractions backend through the extension, or `ReactantEx()`
under Reactant. The executor is a keyword of the domain constructor:

```@example domains
dthreads = StructuredDomain((Pt(100),); U, V, executor = ThreadsEx())
visibilitymap(Gaussian(2.0), dthreads) ≈ vis
```

Inside a Reactant trace the executor is switched to `ReactantEx()` automatically.
