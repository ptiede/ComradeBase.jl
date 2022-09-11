abstract type AbstractIntensityMap{T,S} <: AbstractMatrix{T} end


#abstract type AbstractPolarizedMap{I,Q,U,V} end
"""
    intensitymap(model::AbstractModel, fov, dims; phasecenter = (0.0,0.0), executor=SequentialEx(), pulse=DeltaPulse())

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap`
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
                              fov::NTuple{2},
                              dims::Dims{2};
                              phasecenter = (0.0, 0.0),
                              pulse=ComradeBase.DeltaPulse(),
                              executor=SequentialEx()) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, fov, dims; phasecenter, pulse, executor)
end

function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, args...; kwargs...)
    return intensitymap(s, (fovx, fovy), (ny, nx), args...; kwargs...)
end

"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::AbstractIntensityMap, s::M, executor=SequentialEx()) where {M}
    return intensitymap!(imanalytic(M), img, s, executor)
end

function intensitymap(::IsAnalytic, s,
                      fov::NTuple{2},
                      dims::Dims{2};
                      phasecenter = (0.0, 0.0),
                      pulse=ComradeBase.DeltaPulse(),
                      executor=SequentialEx())
    T = typeof(intensity_point(s, 0.0, 0.0))
    ny, nx = dims
    img = IntensityMap(zeros(T, ny, nx), fov, convert.(Ref(T), phasecenter), pulse)
    intensitymap!(IsAnalytic(), img, s, executor)
    return img
end

function intensitymap!(::IsAnalytic, img::AbstractIntensityMap, s, executor=SequentialEx())
    x,y = imagepixels(img)
    dx, dy = pixelsizes(img)
    @floop executor for I in CartesianIndices(img)
        iy,ix = Tuple(I)
        img[I] = intensity_point(s, x[ix], y[iy])*dx*dy
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
