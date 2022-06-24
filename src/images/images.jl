abstract type AbstractIntensityMap{T,S} <: AbstractMatrix{T} end


#abstract type AbstractPolarizedMap{I,Q,U,V} end
"""
    intensitymap(model::AbstractModel, fovx, fovy, nx, ny; executor=SequentialEx(), pulse=DeltaPulse())

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap`
object that have a field of view of `fovx, fovy` in the x and y direction  respectively
with `nx` pixels in the x-direction and `ny` pixels in the y-direction.

Optionally the user can specify the `pulse` function that converts the image from a discrete
to continuous quantity, and the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap(s::M,
                              fovx::Number, fovy::Number,
                              nx::Int, ny::Int;
                              pulse=ComradeBase.DeltaPulse(),
                              executor=SequentialEx()) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, fovx, fovy, nx, ny; pulse, executor)
end

"""
    intensitymap!(img::AbstractIntensityMap, model, fovx, fovy, nx, ny; executor, pulse)

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::AbstractIntensityMap, s::M, executor=SequentialEx()) where {M}
    return intensitymap!(imanalytic(M), img, s, executor)
end

function intensitymap(::IsAnalytic, s,
                      fovx::Number, fovy::Number,
                      nx::Int, ny::Int;
                      pulse=ComradeBase.DeltaPulse(),
                      executor=SequentialEx())
    T = typeof(intensity_point(s, 0.0, 0.0))
    img = IntensityMap(zeros(T, ny, nx), fovx, fovy, pulse)
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
