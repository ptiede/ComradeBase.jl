# In this file we will define our base image class. This is entirely based on
# AxisKeys.jl. The specialization occurs in the keys themselves for which we define
# a special type. This type is coded to work similar to a Tuple.
const NDA = AxisKeys.NamedDims.NamedDimsArray


abstract type AbstractGrid{N, T} <: AbstractVector{T} end

"""
    $(TYPEDEF)
This struct holds the dimensions that the EHT expect. The first type parameter `N`
defines the names of each dimension. These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
# Fieldnames
$(FIELDS)
# Notes
Warning it is rare you need to access this constructor directly. Instead
use the direct [`IntensityMap`](@ref) function.
"""
struct GriddedKeys{N, G, Hd, T} <: AbstractGrid{N, T}
    dims::G
    header::Hd
end

dims(g::AbstractGrid) = g.dims
header(g::AbstractGrid) = g.header
Base.keys(::AbstractGrid{N}) where {N} = N


Base.getindex(d::AbstractGrid, i::Int) = getindex(dims(d), i)
Base.getindex(d::AbstractGrid, i::Tuple) = getindex(dims(d), i)
Base.length(d::AbstractGrid) = length(dims(d))
Base.firstindex(d::AbstractGrid) = 1
Base.lastindex(d::AbstractGrid) = length(d)
Base.axes(d::AbstractGrid) = axes(dims(d))
Base.iterate(d::AbstractGrid, i::Int = 1) = iterate(dims(d), i)
Base.map(f, d::AbstractGrid) = rebuild(typeof(d), map(f, dims(d)), header(d))
Base.front(d::AbstractGrid) = Base.front(dims(d))
Base.eltype(d::AbstractGrid) = Base.elype(dims(d))

AxisKeys.findindex(sel, axk::AbstractGrid) = AxisKeys.findindex(sel, dims(axk))


Base.getproperty(g::GriddedKeys{Na}, p::Symbol) where {Na} = dims(g)[findfirst(==(p), Na)]

"""
    GriddedKeys{Na}(dims::Tuple, header=nothing) where {Na}

Builds the EHT image dimensions using the names `Na` and dimensions `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.
The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
# Notes
Instead use the direct [`IntensityMap`](@ref) function.
```julia
dims = GriddedKeys{(:X, :Y, :T, :F)}((-5.0:0.1:5.0, -4.0:0.1:4.0, [1.0, 1.5, 1.75], [230, 345]))
```
"""
@inline function GriddedKeys{Na}(dims::Tuple, header=nothing) where {Na}
    @assert length(Na) == length(dims) "The length of names has to equal the number of dims"
    return GriddedKeys{Na, typeof(dims), typeof(header), eltype(first(dims))}(dims, header)
end

"""
    GriddedKeys(dims::NamedTuple{Na}, header=nothing)

Builds the EHT image dimensions using the names `Na` and dimensions are the values of `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.
The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
# Notes
Instead use the direct [`IntensityMap`](@ref) function.
```julia
dims = GriddedKeys((X=-5.0:0.1:5.0, Y=-4.0:0.1:4.0, T=[1.0, 1.5, 1.75], F=[230, 345]))
```
"""
@inline function GriddedKeys(nt::NamedTuple{N}, header=nothing) where {N}
    dims = values(nt)
    return GriddedKeys{N}(dims, header)
end

function rebuild(::Type{<:GriddedKeys{N}}, g, args...) where {N}
    GriddedKeys{N}(g, args...)
end






# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}, <:NamedTuple{(:X,:Y)}}
const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}

const SpatialDataArr = NDA{(:X, :Y)}


# # Our image will be some KeyedArray but where we require specific keys names
const IntensityMap{T,N, G} = KeyedArray{T,N,<:DataArr, G <: AbstractGrid} where {T, N, G}
const SpatialIntensityMap{T,2,G} = KeyedArray{T,2,<:SpatialDataArr, G<:AbstractGrid} where {T,G}


function KeyedArray(data::AbstractArray{T, N}, g::AbstractGrid{Na}) where {T, N, Na}
    narr = NamedDimsArray(data, Na)
    KeyedArray{T, N, typeof(narr), typeof(g)}(narr, g)
end

"""
    IntensityMap(data::AbstractArray, dims::NamedTuple)
    IntensityMap(data::AbstractArray, grid::AbstractGrid)

Constructs an intensitymap using the image dimensions given by `dims`. This returns a
`KeyedArray` with keys given by an `ImageDimensions` object.

```julia
dims = (X=range(-10.0, 10.0, length=100), Y = range(-10.0, 10.0, length=100),
        T = [0.1, 0.2, 0.5, 0.9, 1.0], F = [230e9, 345e9]
        )
imgk = IntensityMap(rand(100,100,5,1), dims)
```
"""
function IntensityMap(data::AbstractArray{T,N}, dims::AbstractGrid) where {T,N}
    AxisKeys.construction_check(data, values(dims))
    return KeyedArray(data, dims)
end

IntensityMap(data::AbstractArray, dims::DataNames, header=nothing) = IntensityMap(data, GriddedKeys(dims, header))




function IntensityMap(data::AbstractArray, fovx::Real, fovy::Real, x0::Real=0.0, y0::Real=0.0)
    X, Y = imagepixels(fovx, fovy, size(data)..., x0, y0)
    return IntensityMap(data, (;X, Y))
end

function _build_named_dims(data, ::NamedTuple{Na}) where {Na}
    return NamedDimsArray(data, Na)
end
