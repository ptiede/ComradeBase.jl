"""
    domainpoints(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is nonallocating
since it lazily computes the grid.
This is useful for broadcasting a model across an abritrary grid.
"""
domainpoints(img::IntensityMap) = domainpoints(axisdims(img))


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
    imagepixels(fovx, fovy, nx, ny, x0=0, y0=0; mdims=(), posang=0.0, executor=Serial(), header=NoHeader())

Construct a spatial grid of pixels with a field of view `fovx` and `fovy` and `nx` and `ny` pixels.
The points are the pixel centers and the field of view goes from the edge of the first pixel
to the edge of the last pixel. The `x0`, `y0` offsets shift the image origin over by
(`x0`, `y0`) in the image plane. 

Additional dimensions (time and/or frequency) are added via `mdims`: a tuple of domain lists. 
- A frequency list is created with `Fr([...])`
- A time list is created with `Ti([...])`
These dimensions are appended to the spatial grid after X and Y.
The dimension ordering in `mdims` determines the ordering of the additional dimensions in the multidomain cube.
X and Y are always the first two dimensions, respectively.

## Arguments:
 - `fovx::Number`: The field of view in the x-direction
 - `fovy::Number`: The field of view in the y-direction
 - `nx::Integer`: The number of pixels in the x-direction
 - `ny::Integer`: The number of pixels in the y-direction

## Keyword Arguments:
 - `x0::Number=0`: The x-offset of the image
 - `y0::Number=0`: The y-offset of the image
 - `mdims::Union{NamedTuple, Tuple}=()` : The non-spatial dimensions of the image (frequency and/or time)
 - `posang::Number=0`: The position angle of the grid, relative to RA=0 axis.
 - `executor=Serial()`: The executor to use for the grid, default is serial execution
 - `header=NoHeader()`: The header to use for the grid

```julia
# create a square 64x64 grid with a FOV of 250μas
julia> grid = imagepixels(μas2rad(250), μas2rad(250), 64, 64)

# create a square 64x64 multidomain grid with a FOV of 250μas
julia> Frlist = Fr([230e9, 345e9])
julia> Tilist = Ti([1, 2, 3])

# multifrequency grid
julia> fr_grid = imagepixels(μas2rad(250), μas2rad(250), 64, 64; mdims=(Frlist, ))

# set index ordering as (X,Y,Fr,Ti)
julia> fr_ti_grid = imagepixels(μas2rad(250), μas2rad(250), 64, 64; mdims=(Frlist, Tilist))

# set index ordering as (X,Y,Ti,Fr)
julia> ti_fr_grid = imagepixels(μas2rad(250), μas2rad(250), 64, 64; mdims=(Tilist, Frlist))
```
"""
function imagepixels(
        fovx::Real, fovy::Real, nx::Integer, ny::Integer,
        x0::Number = zero(fovx), y0::Number = zero(fovy);
        mdims::Union{NamedTuple, Tuple} = (),
        posang::Number = zero(fovx),
        executor = Serial(), header = NoHeader()
    )
    @assert (nx > 0) && (ny > 0) "Number of pixels must be positive"

    psizex = fovx / nx
    psizey = fovy / ny

    xitr = X(LinRange(-fovx / 2 + psizex / 2 - x0, fovx / 2 - psizex / 2 - x0, nx))
    yitr = Y(LinRange(-fovy / 2 + psizey / 2 - y0, fovy / 2 - psizey / 2 - y0, ny))
    grid = RectiGrid((xitr, yitr, mdims...); executor, header, posang)
    return grid
end

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

Computes the flux of a intensity map. For a [`StokesMap`](@ref) the result is a
`StokesParams` holding the flux of each Stokes component.
"""
flux(im::RectiMap) = _flux(im, eldims(im))

_flux(im, ::Tuple{}) = sum(im; dims = (:X, :Y))
function _flux(im, ::Tuple{Stokes})
    return StokesParams(flux(stokes(im, :I)), flux(stokes(im, :Q)), flux(stokes(im, :U)), flux(stokes(im, :V)))
end

flux(im::SpatialIntensityMap) = sum(parent(im))

"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.

For polarized maps we return the centroid for Stokes I only.
"""
centroid(im::RectiMap{<:Real}) = _centroid(im, eldims(im))

_centroid(im, ::Tuple{Stokes}) = centroid(stokes(im, :I))
function _centroid(im, ::Tuple{})
    (; X, Y) = named_dims(im)
    return mapslices(x -> centroid(IntensityMap(x, RectiGrid((; X, Y)))), im; dims = (:X, :Y))
end

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
    return _second_moment(im, eldims(im); center)
end

_second_moment(im, ::Tuple{Stokes}; center) = second_moment(stokes(im, :I); center)
function _second_moment(im, ::Tuple{}; center)
    (; X, Y) = named_dims(im)
    return mapslices(
        x -> second_moment(IntensityMap(x, RectiGrid((; X, Y))); center), im;
        dims = (:X, :Y)
    )
end

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
