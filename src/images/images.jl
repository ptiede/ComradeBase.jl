
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}}
const SpatialOnly = NamedTuple{(:X, :Y)}
using NamedDims
struct ImageDimensions{N,T,H}
    dims::T
    header::H
end

function ImageDimensions(nt::NamedTuple{N}, header=nothing) where {N}
    dims = Tuple(nt)
    return ImageDimensions{N, typeof(dims), typeof(header)}(dims, header)
end


dims(d::ImageDimensions) = d.dims

# Make ImageDimensions act like a tuple
Base.getindex(d::ImageDimensions, i) = getindex(dims(d), i)
Base.length(d::ImageDimensions) = length(d.dims)
Base.firstindex(d::ImageDimensions) = 1
Base.lastindex(d::ImageDimensions) = length(d)
Base.axes(d::ImageDimensions) = axes(d.dims)
Base.iterate(t::ImageDimensions, i::Int=1) = iterate(t.dims, i)
Base.map(f, d::ImageDimensions) = map(f, d.dims)
Base.front(d::ImageDimensions) = Base.front(d.dims)
Base.eltype(d::ImageDimensions) = eltype(d.dims)

RectiGrids.grid(k::KeyedArray{T,N,A,<:ImageDimensions}) where {T,N,A} = grid(named_axiskeys(k))

const IntensityMap{T,N,Na} = KeyedArray{T,N,<:AbstractArray{T,N}, <:ImageDimensions{Na}}

function AxisKeys.KeyedArray(data::AbstractArray{T,N}, keys::ImageDimensions{Na}) where {T,N,Na}
    AxisKeys.construction_check(data, keys.dims)
    a = NamedDimsArray(data, Na)
    return KeyedArray{T,N,typeof(a), typeof(keys)}(data, keys)
end


function IntensityMap(data::AbstractArray, dims::NamedTuple, header=nothing)
    deht = ImageDimensions(dims, header)
    return KeyedArray(data, deht)
end



"""
    intensitymap(model::AbstractModel, fov, dims; phasecenter = (0.0,0.0), executor=SequentialEx(), pulse=DeltaPulse())

Computes the intensity map or _image_ of the `model`. This returns an `DimArray`
object that have a field of view of `fov` where the first element is in the x direction
and the second in the y. The image viewed as a matrix will have dimension `dims` where
the first element is the number of rows or _pixels in the y direction_ and the second
is the number of columns for _pixels in the x direction_.
"""
@inline function intensitymap(s::M,
                              dims::NamedTuple, header=nothing
                              ) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims)
end

function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real, y0::Real)
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"
    psizex=fovx/nx
    psizey=fovy/ny

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)

    return (X=xitr, Y=yitr)
end

function pixelsizes(img::KeyedArray)
    keys = named_axiskeys(img)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end

function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0; frequency=0.0:0.0, time=0.0:0.0, kwargs...)
    X, Y = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, X, Y, Ti(time), Fq(frequency); kwargs...)
end


"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::KeyedArray, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end

function intensitymap(::IsAnalytic, s,
                      dims::NamedTuple{N}, header=nothing) where {N}
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), grid(dims)).*dx.*dy
    return KeyedArray(img, ImageDimensions(dims, header))
end

using RectiGrids

function intensitymap!(::IsAnalytic, img::KeyedArray, s, header=nothing)
    dx, dy = pixelsizes(img)
    g = grid(named_axiskeys(img))
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return KeyedArray(img, dims)
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



include("polarizedtypes.jl")
include("intensitymap.jl")
#include("polarizedmap.jl")
