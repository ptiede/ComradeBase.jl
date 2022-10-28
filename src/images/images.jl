"""
    GriddedPositions(grids...)

A grid that contains all the gridded information for an image.
This is usually built on `DimensionalData`'s `Dim` types which
gives the meaning to each grid.
"""
struct GriddedPositions{G}
    grids::G
end


"""
    intensitymap(model::AbstractModel, fov, dims; phasecenter = (0.0,0.0), executor=SequentialEx(), pulse=DeltaPulse())

Computes the intensity map or _image_ of the `model`. This returns an `DimArray`
object that have a field of view of `fov` where the first element is in the x direction
and the second in the y. The image viewed as a matrix will have dimension `dims` where
the first element is the number of rows or _pixels in the y direction_ and the second
is the number of columns for _pixels in the x direction_.

# Warning
Note that the order of fov and dims are switched.

# Keywords
Optionally the user can specify the:
    - `phasecenter` the offset from the center of the image that we define as the origin
    - `pulse` function that converts the image from a discrete
to continuous quantity
    - `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap(s::M,
                              dims...,
                              ;executor=SequentialEx()) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims...; executor)
end

function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real, y0::Real)
    psizex=fovx/max(nx,1)
    psizey=fovy/max(ny,1)

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)

    return X(xitr), Y(yitr)
end

function pixelsizes(img::DimArray)
    d = dims(img)
    n = name.(d)
    return NamedTuple{n}(step.(d))
end

function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0; kwargs...)
    X, Y = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, X, Y; kwargs...)
end

"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::Union{AbstractDimArray, AbstractDimStack}, s::M; executor=SequentialEx()) where {M}
    return intensitymap!(imanalytic(M), img, s, executor)
end

function intensitymap(::IsAnalytic, s,
                      dims...;
                      executor=SequentialEx())
    T = typeof(intensity_point(s, 0.0, 0.0))
    img = DimArray(zeros(T, dims...), dims)
    intensitymap!(IsAnalytic(), img, s, executor)
    return img
end

function intensitymap!(::IsAnalytic, img::Union{AbstractDimArray, AbstractDimStack}, s, executor=SequentialEx())
    dx, dy = pixelsizes(img)
    n = name.(dims(img))
    @floop executor for p in DimPoints(img)
        np = NamedTuple{n}(p)
        img[I] = intensity_point(s, np)*dx*dy
    end
    return img
end


# function intensitymap(::IsAnalytic, s, fovx::Number, fovy::Number, nx::Int, ny::Int; pulse=ComradeBase.DeltaPulse())
#     x,y = imagepixels(fovx, fovy, nx, ny)
#     pimg = map(CartesianIndices((1:ny,1:nx))) do I
#         iy,ix = Tuple(I)
#         @inbounds f = intensity_point(s, x[ix], y[iy])
#         return f
#     end
#     return IntensityMap(pimg, fovx, fovy, pulse)
# end



# function intensitymap!(::IsAnalytic, im::AbstractIntensityMap, m)
#     xitr, yitr = imagepixels(im)
#     @inbounds for (i,x) in pairs(xitr), (j,y) in pairs(yitr)
#         im[j, i] = intensity_point(m, x, y)
#     end
#     return im
# end



include("pulse.jl")
include("polarizedtypes.jl")
include("intensitymap.jl")
#include("polarizedmap.jl")
