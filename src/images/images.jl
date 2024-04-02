export IntensityMap, SpatialIntensityMap,
        DataArr, SpatialDataArr, DataNames,
        named_axisdims, imagepixels, pixelsizes, imagegrid,
        phasecenter, baseimage
include("grid.jl")
include("dim_image.jl")

export StokesIntensityMap, stokes
include("stokes_image.jl")

const IntensityMapTypes{T,N} = Union{IntensityMap{T,N}, StokesIntensityMap{T,N}}

export flux, centroid, second_moment, named_axisdims, axisdims,
       imagepixels, pixelsizes, imagegrid, phasecenter
include("methods.jl")
include("io.jl")
include("rrules.jl")


"""
    intensitymap(model::AbstractModel, dims::AbstractGrid)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `IntensityMap` with `dims` an [`AbstractGrid`](@ref) as dimensions.
"""
@inline function intensitymap(s::M,
                              dims::AbstractGrid
                              ) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims)
end
@inline intensitymap(::IsAnalytic, m::AbstractModel, dims::AbstractGrid)  = intensitymap_analytic(m, dims)
@inline intensitymap(::NotAnalytic, m::AbstractModel, dims::AbstractGrid) = intensitymap_numeric(m, dims)


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
@inline intensitymap!(::IsAnalytic, img::IntensityMapTypes, m::AbstractModel)  = intensitymap_analytic!(img, m)
@inline intensitymap!(::NotAnalytic, img::IntensityMapTypes, m::AbstractModel) = intensitymap_numeric!(img, m)


@inline intensitymap(s::AbstractModel, dims::NamedTuple) = intensitymap(s, RectiGrid(dims))

function intensitymap_analytic(s::AbstractModel, dims::AbstractGrid)
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), imagegrid(dims)).*dx.*dy
    return IntensityMap(img, dims)
end


function intensitymap_analytic!(img::IntensityMapTypes, s)
    dx, dy = pixelsizes(img)
    g = imagegrid(img)
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return img
end


"""
    intensitymap(s, fovx, fovy, nx, ny, x0=0.0, y0=0.0)

Creates a *spatial only* IntensityMap intensity map whose pixels in the `x`, `y` direction are
such that the image has a field of view `fovx`, `fovy`, with the number of pixels `nx`, `ny`,
and the origin or phase center of the image is at `x0`, `y0`.
"""
function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0)
    grid = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, grid)
end
