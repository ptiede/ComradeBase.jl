# In this file we will define our base image class. This is entirely based on
# AxisKeys.jl. The specialization occurs in the keys themselves for which we define
# a special type. This type is coded to work similar to a Tuple.
const NDA = AxisKeys.NamedDims.NamedDimsArray

# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}, <:NamedTuple{(:X,:Y)}}
const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}

const SpatialDataArr = NDA{(:X, :Y)}


# # Now we define our custom image type which is a parametric KeyedArray with a specific key type.
const IntensityMap{T,N} = KeyedArray{T,N,<:DataArr}
const SpatialIntensityMap{T,N} = KeyedArray{T,2,<:SpatialDataArr}

"""
    grid(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is unallocating
since it lazily computes the grid. The grid is an example of a KeyedArray and works similarly.
This is useful for broadcasting a model across an abritrary grid.
"""
imagegrid(img::IntensityMap) = grid(named_axiskeys(img))

imagegrid(dims::DataNames) = grid(dims)


"""
    IntensityMap(data::AbstractArray, dims::NamedTuple)

Constructs an intensitymap using the image dimensions given by `dims`. This returns a
`KeyedArray` with keys given by an `ImageDimensions` object.

```julia
dims = (X=range(-10.0, 10.0, length=100), Y = range(-10.0, 10.0, length=100),
        T = [0.1, 0.2, 0.5, 0.9, 1.0], F = [230e9, 345e9]
        )
imgk = IntensityMap(rand(100,100,5,1), dims)
```
"""
function IntensityMap(data::AbstractArray{T,N}, dims::DataNames) where {T,N}
    a = _build_named_dims(data, dims)
    AxisKeys.construction_check(data, values(dims))
    a = NamedDimsArray(data, keys(dims))
    return KeyedArray{T,N,typeof(a), typeof(values(dims))}(data, values(dims))
end

function IntensityMap(data::AbstractArray, fovx::Real, fovy::Real)
    X, Y = imagepixels(fovx, fovy, size(data)..., 0.0, 0.0)
    return IntensityMap(data, (;X, Y))
end

function _build_named_dims(data, ::NamedTuple{Na}) where {Na}
    return NamedDimsArray(data, Na)
end


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

function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real, y0::Real)
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"

    psizex=fovx/nx
    psizey=fovy/ny

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)

    return (X=xitr, Y=yitr)
end

function phasecenter(img::IntensityMap)
    x0 = median(img.X)
    y0 = median(img.Y)
    return (X=x0, Y=y0)
end

imagepixels(img::IntensityMap) = (X=img.X, Y=img.Y)

function fieldofview(img::IntensityMap)
    return fieldofview(named_axiskeys(img))
end

function fieldofview(dims::DataNames)
    (;X,Y) = dims
    dx = step(X)
    dy = step(Y)
    (X=abs(last(X) - first(X))+dx, Y=abs(last(Y)-first(Y))+dy)
end

function pixelsizes(img::IntensityMap)
    keys = imagepixels(img)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end

"""
    intensitymap(s, fovx, fovy, nx, ny, x0=0.0, y0=0.0)
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
@inline function intensitymap!(img::IntensityMap, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end

function intensitymap(::IsAnalytic, s, dims::DataNames)
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), imagegrid(dims)).*dx.*dy
    return img
end


function intensitymap!(::IsAnalytic, img::IntensityMap, s)
    dx, dy = pixelsizes(img)
    g = imagegrid(img)
    parent(img) .= intensity_point.(Ref(s), g).*dx.*dy
    return img
end
