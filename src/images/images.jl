
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}}
const SpatialOnly = NamedTuple{(:X, :Y)}
using NamedDims
struct ImageDimensions{N,T,H}
    dims::T
    header::H
end

function ImageDimensions{Na}(dims::Tuple, header=nothing) where {Na}
    @assert length(Na) == length(dims) "The length of names has to equal the number of dims"
    return ImageDimensions{Na, typeof(dims), typeof(header)}(dims, header)
end

function ImageDimensions(nt::NamedTuple{N}, header=nothing) where {N}
    dims = Tuple(nt)
    return ImageDimensions{N, typeof(dims), typeof(header)}(dims, header)
end

const IntensityMap{T,N,Na} = KeyedArray{T,N,<:AbstractArray{T,N}, <:ImageDimensions{Na}}


dims(d::ImageDimensions) = d.dims
# AxisKeys.getkey(A::IntensityMap; kw...)

# Make ImageDimensions act like a tuple
Base.getindex(d::ImageDimensions, i::Int) = getindex(dims(d), i)
Base.length(d::ImageDimensions) = length(d.dims)
Base.firstindex(d::ImageDimensions) = 1
Base.lastindex(d::ImageDimensions) = length(d)
Base.axes(d::ImageDimensions) = axes(d.dims)
Base.iterate(t::ImageDimensions, i::Int=1) = iterate(t.dims, i)
Base.map(f, d::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(map(f, d.dims), d.header)
Base.front(d::ImageDimensions) = Base.front(d.dims)
Base.eltype(d::ImageDimensions) = eltype(d.dims)
AxisKeys.axiskeys(d::IntensityMap) = getfield(d, :keys)
AxisKeys.axiskeys(x::IntensityMap, d::Int) = d<=ndims(x) ? getindex(axiskeys(x), d) : OneTo(1)

AxisKeys.unify_longest(x::ImageDimensions) = x
AxisKeys.unify_longest(x::ImageDimensions{Na}, y::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(dims(x), dims(y)), x.header)
AxisKeys.unify_longest(x::ImageDimensions{Na}, y::Tuple) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(dims(x), y), x.header)
AxisKeys.unify_longest(x::Tuple, y::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(x, dims(y)), y.header)

for (get_or_view, key_get, maybe_copy) in [
    (:getindex, :(AxisKeys.keys_getindex), :copy),
    (:view, :(AxisKeys.keys_view), :identity)
    ]
    @eval begin
        function Base.$get_or_view(A::IntensityMap, raw_inds...)
            inds = to_indices(A, raw_inds)
            @boundscheck checkbounds(parent(A), inds...)
            data = @inbounds $get_or_view(parent(A), inds...)
            inds isa Tuple{Vararg{Integer}} && return data # scalar output

            raw_keys = $key_get(axiskeys(A), inds)
            raw_keys === () && return data # things like A[A .> 0]

            new_keys = ntuple(ndims(data)) do d
                raw_keys === nothing && return axes(data, d)
                raw_keys[d]
            end
            KeyedArray(data, ImageDimensions{dimnames(data)}(new_keys, getfield(A,:keys).header))

        end

        @inline function $key_get(keys::ImageDimensions, inds::Tuple{AbstractVector, Vararg{Any}})
            $key_get(keys.dims, inds)
        end
    end
end




RectiGrids.grid(k::IntensityMap) = grid(named_axiskeys(k))


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


function intensitymap!(::IsAnalytic, img::KeyedArray, s, header=nothing)
    dx, dy = pixelsizes(img)
    g = grid(named_axiskeys(img))
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return KeyedArray(img, dims)
end

"""
    flux(im::AbstractDimArray)

Computes the flux of a intensity map
"""
function flux(im::IntensityMap{T,N}) where {T,N}
    return sum(im, dims=(:X, :Y))
end

flux(im::IntensityMap{T,2}) where {T} = sum(im)



"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.
"""
function centroid(im::IntensityMap{T,N, <:DataNames}) where {T,N}
    return mapslices(centroid, im; dims=(:X, :Y))
end

function centroid(im::IntensityMap{T,2}) where {T}
    for i in axes(im,:X), j in axes(im,:Y)
        x0 += xitr[i].*im[X=i, Y=j]
        y0 += yitr[j].*im[X=I, Y=j]
    end
    return x0/f, y0/f
end

"""
    inertia(im::AbstractIntensityMap; center=true)

Computes the image inertia aka **second moment** of the image.
By default we really return the second **cumulant** or second centered
second moment, which is specified by the `center` argument.
"""
function inertia(im::IntensityMap; center=true)
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    xitr, yitr = imagepixels(im)
    for i in axes(img, :X) j in axes(img, :Y)
        x = xitr[i]
        y = yitr[j]
        xx += x^2*im[j,i]
        yy += y^2*im[j,i]
        xy += x*y*im[j,i]
    end

    if center
        x0, y0 = centroid(im)
        xx -= x0^2
        yy -= y0^2
        xy -= x0*y0
    end

    return @SMatrix [xx/f xy/f; xy/f yy/f]
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
