# Domains and maps

```@meta
CurrentModule = ComradeBase
```

A domain says where a model is evaluated. ComradeBase has two kinds:

- [`RectiGrid`](@ref): a rectilinear image grid with spatial dims `X`, `Y` and optional
  `Ti` (time) and `Fr` (frequency) dims.
- [`StructuredDomain`](@ref): a set of points, such as the baselines of a VLBI observation,
  with an index dim `Node` and optional plane dims.

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
dim is always `Node`, the point index; `Node(n)` is shorthand for `Node(Base.OneTo(n))`. A
domain with only `Node` is a list of points:

```@example domains
U = 0.1 .* randn(100)
V = 0.1 .* randn(100)
d = StructuredDomain((Node(100),); U, V)
vis = visibilitymap(Gaussian(2.0), d)
size(vis)
```

`UnstructuredDomain((; U, V))` and `StructuredDomain((; U, V))` build the same `(Node,)`
domain from a `NamedTuple`.

Plane dims such as `Fr` follow `Node`. Each coordinate array spans a subset of the dims,
in the order they appear in the domain, and its axes must match those dims. The span is
inferred from the array's size when only one ordered subset of the dims matches it;
otherwise it is given as `array => (:Node, :Fr)`.

Here each row has a measurement at every frequency. Baselines are given in meters as `u`,
`v`, so they vary with `Node` only and are converted to wavelengths with the `Fr` lookup
(in Hz). A per-point observation time `Ti` spans `Node`:

```@example domains
npt = 50
dfr = StructuredDomain(
    (Node(npt), Fr([230.0e9, 345.0e9]));
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

A map over a `StructuredDomain` is indexed by dim name. Slicing along `Node` or `Fr` slices the
coordinates with the values:

```@example domains
visfr = visibilitymap(Gaussian(2.0), dfr)
sub = visfr[Node(1:10), Fr(2)]
size(sub), size(axisdims(sub).U)
```

### Movies and frame assignment

Points are not grouped by frame. A movie's points stay in one `(Node,)` or `(Node, Fr)`
domain, and the `Ti` coordinate records each point's observation time. The image side
of a movie is a grid with a `Ti` dim; the step that computes visibilities from the image
planes looks up each point's frame from its `Ti` coordinate. A `(Node, Ti, Fr)` domain can
be constructed, but nothing requires one.

[`frameindex`](@ref) does that lookup. For a dim with `Intervals` sampling, such as one
built by [`frames`](@ref), a point belongs to the interval that contains it; otherwise
its coordinate must equal a plane's value. A point that matches no plane is an error.

```@example domains
scans = frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])
gmovie = spatialgrid(20.0, 20.0, 32, 32) ⊗ scans
frameindex(scans, [0.2, 2.0, 4.9])
```

### Choosing a layout

A dim of the domain after `Node` is worth having only when the same points repeat along it.
Anything that varies from point to point, such as the baselines present in a scan, goes
into `Node` with a per-point coordinate.

**Scans with different baselines.** VLBI coverage changes from scan to scan, so all scans
share one `Node` dim and `Ti` records each point's scan time. Each scan can have any number
of baselines. Every baseline here is measured in both bands, so `Fr` is a dim:

```@example domains
nbl = [10, 3, 6]                       # baselines in each scan
t = reduce(vcat, fill.([0.1, 1.7, 4.2], nbl))
n = length(t)
dscans = StructuredDomain(
    (Node(n), Fr([227.1e9, 229.1e9]));
    u = 4.0e6 .* randn(n), v = 4.0e6 .* randn(n), Ti = t,
)
frameindex(scans, dscans.Ti)
```

Keep the points sorted by time. Nothing requires it for evaluation, but each frame's points
are then contiguous.

**Epochs months apart.** For a time lapse in which every track is one frame, give each track
its own interval. The gaps between tracks need no frames, and a point outside every track is
an error:

```@example domains
tracks = frames(Ti, [59000.0, 59030.0, 59095.0], [59000.5, 59030.5, 59095.5])  # MJD
frameindex(tracks, [59000.2, 59030.4, 59095.1])
```

Labeling the points with an epoch number instead matches them by equality, with no interval
bounds to choose:

```@example domains
frameindex(Ti([1, 2, 3]), [1, 1, 3, 2])
```

**Missing channels.** An `Fr` dim says that every point is measured at every frequency. When
some baselines lack some channels, give each point its own `Fr` coordinate; image planes
are then matched to it like times:

```@example domains
dch = UnstructuredDomain((; U = 0.1 .* randn(4), V = 0.1 .* randn(4), Fr = [230.0e9, 230.0e9, 345.0e9, 230.0e9]))
frameindex(Fr([230.0e9, 345.0e9]), dch.Fr)
```

**A time dim on the data.** A `(Node, Ti)` domain needs the same number of points in every
frame. With uneven scans, `Node` has the length of the largest scan and the `valid`
coordinate marks the padding. Evaluation does not read `valid`: padded entries are evaluated
like any other, so give them finite coordinates, and leave them out when comparing to data.
This wastes the padded points and is rarely needed:

```@example domains
nmax = maximum(nbl)
mask = [i <= nbl[k] for i in 1:nmax, k in eachindex(nbl)]
dpad = StructuredDomain(
    (Node(nmax), Ti([0.1, 1.7, 4.2]));
    U = 0.1 .* randn(nmax, 3) .* mask, V = 0.1 .* randn(nmax, 3) .* mask, valid = mask,
)
ComradeBase.coordspans(dpad)
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
dthreads = StructuredDomain((Node(100),); U, V, executor = ThreadsEx())
visibilitymap(Gaussian(2.0), dthreads) ≈ vis
```

Inside a Reactant trace the executor is switched to `ReactantEx()` automatically.
