 export IntensityMap, SpatialIntensityMap,
        DataArr, SpatialDataArr, DataNames,
        named_axiskeys, imagepixels, pixelsizes, imagegrid,
        phasecenter, baseimage
 include("keyed_image.jl")

export StokesIntensityMap, stokes
include("stokes_image.jl")

const IntensityMapTypes{T,N} = Union{IntensityMap{T,N}, StokesIntensityMap{T,N}}

export flux, centroid, second_moment, named_axiskeys, axiskeys,
       imagepixels, pixelsizes, imagegrid, phasecenter
include("methods.jl")


"""
    intensitymap(model::AbstractModel, dims)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `KeyedArray` with [`ImageDimensions`](@ref) as keys. The dimensions are a `NamedTuple`
and must have one of the following names:
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
"""
@inline function intensitymap(s::M,
                              dims::DataNames
                              ) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims)
end


"""
    intensitymap(s, fovx, fovy, nx, ny, x0=0.0, y0=0.0)

Creates a *spatial only* IntensityMap intensity map whose pixels in the `x`, `y` direction are
such that the image has a field of view `fovx`, `fovy`, with the number of pixels `nx`, `ny`,
and the origin or phase center of the image is at `x0`, `y0`.
"""
function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0)
    (;X, Y) = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, (X=X, Y=Y))
end


"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::IntensityMapTypes, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end

function intensitymap(::IsAnalytic, s, dims::DataNames)
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), imagegrid(dims)).*dx.*dy
    return img
end


function intensitymap!(::IsAnalytic, img::IntensityMapTypes, s)
    dx, dy = pixelsizes(img)
    g = imagegrid(img)
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return img
end
