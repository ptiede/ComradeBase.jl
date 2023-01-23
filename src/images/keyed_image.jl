# In this file we will define our base image class. This is entirely based on
# AxisKeys.jl. The specialization occurs in the keys themselves for which we define
# a special type. This type is coded to work similar to a Tuple.
const NDA = AxisKeys.NamedDims.NamedDimsArray
export GriddedKeys, named_dims, dims, header, axisdims

abstract type AbstractDims{N, T} <: AbstractVector{T} end

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
struct GriddedKeys{N, G, Hd, T} <: AbstractDims{N, T}
    dims::G
    header::Hd
end

dims(g::AbstractDims) = getfield(g, :dims)
named_dims(g::AbstractDims{N}) where {N} = NamedTuple{N}(dims(g))
header(g::AbstractDims) = getfield(g, :header)
Base.keys(::AbstractDims{N}) where {N} = N


Base.getindex(d::AbstractDims, i::Int) = getindex(dims(d), i)
Base.getindex(d::AbstractDims, i::Tuple) = getindex(dims(d), i)
Base.length(d::AbstractDims) = length(dims(d))
Base.firstindex(d::AbstractDims) = 1
Base.lastindex(d::AbstractDims) = length(d)
Base.axes(d::AbstractDims) = axes(dims(d))
Base.iterate(d::AbstractDims, i::Int = 1) = iterate(dims(d), i)
Base.map(f, d::AbstractDims) = rebuild(typeof(d), map(f, dims(d)), header(d))
Base.front(d::AbstractDims) = Base.front(dims(d))
Base.eltype(d::AbstractDims) = Base.eltype(dims(d))
AxisKeys.axiskeys(img::IntensityMap) = dims(getfield(img, :keys))


"""
    axisdims(img::IntensityMap)

Returns the keys of the `IntensityMap` as the actual internal `AbstractDims` object.
"""
axisdims(img::IntensityMap) = getfield(img, :keys)

AxisKeys.findindex(sel, axk::AbstractDims) = AxisKeys.findindex(sel, dims(axk))
# AxisKeys.keys_getindex(keys::AbstractDims{N}, inds) where {N} = GriddedKeys{N}(AxisKeys.keys_getindex(dims(keys), inds))

Base.propertynames(::GriddedKeys{Na}) where {Na} = Na
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

function rebuild(::Type{<:GriddedKeys{N}}, g, header=nothing) where {N}
    GriddedKeys{N}(g, header)
end

AxisKeys.unify_longest(x::AbstractDims) = x
AxisKeys.unify_longest(x::AbstractDims{Na}, y::AbstractDims{Na}) where {Na} = rebuild(typeof(x), AxisKeys.unify_longest(dims(x), dims(y)), header(x))
AxisKeys.unify_longest(x::AbstractDims{Na}, y::Tuple) where {Na} = rebuild(typeof(x), AxisKeys.unify_longest(dims(x), y), header(x))
AxisKeys.unify_longest(x::Tuple, y::AbstractDims{Na}) where {Na} = AxisKeys.unify_longest(y, x)

function AxisKeys.unify_keys(left::AbstractDims, right::AbstractDims)
    s = map(AxisKeys.unify_one, dims(left), dims(right))
    return (Tuple(s))
end

function AxisKeys.unify_keys(left::AbstractDims{Na}, right::Tuple) where {Na}
    return rebuild(typeof(left), AxisKeys.unify_keys(dims(left), right))
end

AxisKeys.unify_keys(left::Tuple, right::AbstractDims) = AxisKeys.unify_keys(right, left)





# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)},
                        <:NamedTuple{(:X, :Y, :T)}, <:NamedTuple{(:X, :Y, :F)},
                        <:NamedTuple{(:X,:Y)}}
# const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}

const SpatialDataArr = NDA{(:X, :Y)}


# # Our image will be some KeyedArray but where we require specific keys names
const IntensityMap{T,N,G} = KeyedArray{T,N,<:Any, G} where {T, N, G<:AbstractDims}
const SpatialIntensityMap{T,G} = KeyedArray{T,2,<:Any, G} where {T,G<:AbstractDims}


function KeyedArray(data::AbstractArray{T, N}, g::AbstractDims{Na}) where {T, N, Na}
    narr = NamedDimsArray(data, Na)
    KeyedArray{T, N, typeof(narr), typeof(g)}(narr, g)
end

function KeyedArray(data::AbstractVector{T}, g::AbstractDims{N}) where {T, N}
    narr = NamedDimsArray(data, N)
    return KeyedArray{T, 1, typeof(data), typeof(g)}(narr, g)
end


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
            KeyedArray(data, GriddedKeys{dimnames(data)}(new_keys, header(axiskeys(A))))

        end

        @inline function $key_get(keys::AbstractDims, inds)
            $key_get(dims(keys), inds)
        end
    end
end




"""
    IntensityMap(data::AbstractArray, dims::NamedTuple)
    IntensityMap(data::AbstractArray, grid::AbstractDims)

Constructs an intensitymap using the image dimensions given by `dims`. This returns a
`KeyedArray` with keys given by an `ImageDimensions` object.

```julia
dims = (X=range(-10.0, 10.0, length=100), Y = range(-10.0, 10.0, length=100),
        T = [0.1, 0.2, 0.5, 0.9, 1.0], F = [230e9, 345e9]
        )
imgk = IntensityMap(rand(100,100,5,1), dims)
```
"""
function IntensityMap(data::AbstractArray{T,N}, g::AbstractDims) where {T,N}
    AxisKeys.construction_check(data, dims(g))
    return KeyedArray(data, g)
end

IntensityMap(data::AbstractArray, dims::NamedTuple, header=nothing) = IntensityMap(data, GriddedKeys(dims, header))

function IntensityMap(data::AbstractArray, fovx::Real, fovy::Real, x0::Real=0.0, y0::Real=0.0)
    grid = imagepixels(fovx, fovy, size(data)..., x0, y0)
    return IntensityMap(data, grid)
end
