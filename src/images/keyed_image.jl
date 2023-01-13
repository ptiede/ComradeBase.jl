# In this file we will define our base image class. This is entirely based on
# AxisKeys.jl. The specialization occurs in the keys themselves for which we define
# a special type. This type is coded to work similar to a Tuple.
const NDA = AxisKeys.NamedDims.NamedDimsArray

# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}, <:NamedTuple{(:X,:Y)}}
const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}

const SpatialDataArr = NDA{(:X, :Y)}


# # Our image will be some KeyedArray but where we require specific keys names
const IntensityMap{T,N} = KeyedArray{T,N,<:DataArr} where {T, N}
const SpatialIntensityMap{T,N} = KeyedArray{T,2,<:SpatialDataArr} where {T}



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


function IntensityMap(data::AbstractArray, fovx::Real, fovy::Real, x0::Real=0.0, y0::Real=0.0)
    X, Y = imagepixels(fovx, fovy, size(data)..., x0, y0)
    return IntensityMap(data, (;X, Y))
end

function _build_named_dims(data, ::NamedTuple{Na}) where {Na}
    return NamedDimsArray(data, Na)
end
