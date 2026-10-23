"""
    domainpoints(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is nonallocating
since it lazily computes the grid.
This is useful for broadcasting a model across an abritrary grid.
"""
domainpoints(img::IntensityMap) = domainpoints(axisdims(img))


"""
    basedim(x)

Returns the plain values underneath a dim or lookup, and `x` itself for anything else.
"""
@inline basedim(x::DD.Dimension) = basedim(parent(x))
@inline basedim(x::DD.Lookups.LookupArray) = basedim(parent(x))
@inline basedim(x) = x

"""
    phasecenter(img::IntensityMap)

Computes the phase center of an intensity map. Note this is the pixels that is in
the middle of the image.
"""
function phasecenter(dims::AbstractRectiGrid)
    (; X, Y) = dims
    x0 = -(last(X) + first(X)) / 2
    y0 = -(last(Y) + first(Y)) / 2
    return (X = x0, Y = y0)
end
phasecenter(img::RectiMap) = phasecenter(axisdims(img))

# ChainRulesCore.@non_differentiable pixelsizes(img::IntensityMap)

"""
    spatialgrid(fovx, fovy, nx, ny, x0=0, y0=0; posang=0, executor=Serial(), header=NoHeader())

Constructs the `(X, Y)` grid of `nx × ny` pixels spanning a field of view `fovx × fovy`.
The points are the pixel centers, and the field of view runs from the outer edge of the
first pixel to the outer edge of the last. The image origin is shifted by (`x0`, `y0`).
Non-spatial dims (frequency, time) are appended with [`gridproduct`](@ref), e.g.
`spatialgrid(fov, fov, 64, 64) ⊗ Fr([230e9, 345e9])`.

## Arguments
 - `fovx`, `fovy`: the field of view along `X` and `Y`
 - `nx`, `ny`: the number of pixels along `X` and `Y`
 - `x0`, `y0`: the offset of the image origin

## Keyword Arguments
 - `posang=0`: the position angle of the grid, relative to the RA = 0 axis
 - `executor=Serial()`: the executor of the grid
 - `header=NoHeader()`: the header of the grid

```julia
julia> g = spatialgrid(μas2rad(250), μas2rad(250), 64, 64)

julia> gfr = g ⊗ Fr([230e9, 345e9])                          # dims (X, Y, Fr)

julia> gtifr = g ⊗ Ti([1.0, 2.0, 3.0]) ⊗ Fr([230e9, 345e9])   # dims (X, Y, Ti, Fr)
```
"""
function spatialgrid(
        fovx::Real, fovy::Real, nx::Integer, ny::Integer,
        x0::Number = zero(fovx), y0::Number = zero(fovy);
        posang::Number = zero(fovx),
        executor = Serial(), header = NoHeader()
    )
    (nx > 0 && ny > 0) || throw(ArgumentError("the number of pixels must be positive, got nx = $nx, ny = $ny"))

    psizex = fovx / nx
    psizey = fovy / ny

    xs = LinRange(-fovx / 2 + psizex / 2 - x0, fovx / 2 - psizex / 2 - x0, nx)
    ys = LinRange(-fovy / 2 + psizey / 2 - y0, fovy / 2 - psizey / 2 - y0, ny)
    return RectiGrid((X(_pixellookup(xs, psizex)), Y(_pixellookup(ys, psizey))); executor, header, posang)
end

# A fully specified lookup: `DD.format` cannot infer the traits of a bare range.
_pixellookup(r::AbstractRange, psize) = DD.Lookups.Sampled(
    r; order = DD.Lookups.ForwardOrdered(), span = DD.Lookups.Regular(psize),
    sampling = DD.Lookups.Points()
)

"""
    fieldofview(img::IntensityMap)
    fieldofview(img::IntensityMap)

Returns a named tuple with the field of view of the image.
"""
function fieldofview(img::RectiMap)
    return fieldofview(axisdims(img))
end

pixelsizes(img::RectiMap) = pixelsizes(axisdims(img))

"""
    flux(im::IntensityMap)

Computes the flux of a intensity map: the sum over `X` and `Y`, a map over the other dims if
there are any. For a [`StokesMap`](@ref) the flux of each Stokes component is summed separately
and the result has `StokesParams` elements.
"""
flux(im::RectiMap) = sum(im; dims = (:X, :Y))

flux(im::SpatialIntensityMap{<:Number}) = sum(parent(im))

flux(im::RectiMap{<:StokesParams}) = _stokesflux(map(K -> flux(stokes(im, K)), (:I, :Q, :U, :V)))

_stokesflux(f::NTuple{4, Number}) = StokesParams(f...)
function _stokesflux(f::NTuple{4, IntensityMap})
    data = StructArray{StokesParams{eltype(first(f))}}(map(baseimage, f))
    return IntensityMap(data, axisdims(first(f)), refdims(first(f)), DD.name(first(f)))
end

"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.

For polarized maps we return the centroid for Stokes I only.
"""
function centroid(im::RectiMap{<:Real})
    (; X, Y) = named_dims(im)
    return mapslices(x -> centroid(IntensityMap(x, RectiGrid((; X, Y)))), im; dims = (:X, :Y))
end
centroid(im::StokesMap) = centroid(stokes(im, :I))

function centroid(im::RectiMap{T, 2})::Tuple{T, T} where {T <: Real}
    f = flux(im)
    d = domainpoints(im)
    # Grab the parent otherwise things don't work on the GPU (DD missing multiargument mapreduce)
    cent = mapreduce(+, baseimage(im), d; init = SVector(zero(f), zero(f))) do I, (x, y)
        x0 = x .* I
        y0 = y .* I
        return SVector(x0, y0)
    end
    return cent[1] / f, cent[2] / f
end

"""
    second_moment(im::AbstractIntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.

For polarized maps we return the second moment for Stokes I only.
"""
function second_moment(im::RectiMap{T, N}; center = true) where {T <: Number, N}
    (; X, Y) = named_dims(im)
    return mapslices(
        x -> second_moment(IntensityMap(x, RectiGrid((; X, Y))); center), im;
        dims = (:X, :Y)
    )
end
second_moment(im::StokesMap; center = true) = second_moment(stokes(im, :I); center)

"""
    second_moment(im::IntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.
"""
function second_moment(im::RectiMap{T, 2}; center = true) where {T <: Number}
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    for (I, (x, y)) in pairs(DimPoints(im))
        xx += x .^ 2 * im[I]
        yy += y .^ 2 * im[I]
        xy += x .* y .* im[I]
    end

    if center
        x0, y0 = centroid(im)
        xx = xx ./ f - x0 .^ 2
        yy = yy ./ f - y0 .^ 2
        xy = xy ./ f - x0 .* y0
    else
        xx = xx ./ f
        yy = yy ./ f
        xy = xy ./ f
    end

    return @SMatrix [xx xy; xy yy]
end
