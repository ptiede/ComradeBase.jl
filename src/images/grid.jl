# In this file we will define our base image class. This is entirely based on
export RectiGrid, named_dims, dims, header, axisdims

abstract type AbstractGrid{D} end


abstract type AbstractHeader end

"""
    MinimalHeader{T}

A minimal header type for ancillary image information.

# Fields
$(FIELDS)
"""
struct MinimalHeader{T} <: AbstractHeader
    """
    Common source name
    """
    source::String
    """
    Right ascension of the image in degrees (J2000)
    """
    ra::T
    """
    Declination of the image in degrees (J2000)
    """
    dec::T
    """
    Modified Julian Date in days
    """
    mjd::T
    """
    Frequency of the image in Hz
    """
    frequency::T
end

function MinimalHeader(source, ra, dec, mjd, freq)
    raT, decT, mjdT, freqT = promote(ra, dec, mjd, freq)
    return MinimalHeader(source, raT, decT, mjdT, freqT)
end

"""
    NoHeader


"""
struct NoHeader <: AbstractHeader end


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
struct RectiGrid{D, Hd<:AbstractHeader} <: AbstractGrid{D}
    dims::D
    header::Hd
end

Base.keys(g::RectiGrid) = map(name, dims(g))

@inline RectiGrid(g::RectiGrid) = g

"""
    dims(g::AbstractGrid)

Returns a tuple containing the dimensions of `g`. For a named version see [`ComradeBase.named_dims`](@ref)
"""
DD.dims(g::AbstractGrid) = getfield(g, :dims)

"""
    named_dims(g::AbstractGrid)

Returns a named tuple containing the dimensions of `g`. For a unnamed version see [`dims`](@ref)
"""
named_dims(g::AbstractGrid) = NamedTuple{keys(g)}(dims(g))

"""
    header(g::AbstractGrid)

Returns the headerinformation of the dimensions `g`
"""
header(g::AbstractGrid) = getfield(g, :header)
Base.keys(g::AbstractGrid) = MethodError("You must implement `Base.keys($(typeof(g)))`")

ChainRulesCore.@non_differentiable header(AbstractGrid)


Base.getindex(d::AbstractGrid, i::Int) = getindex(dims(d), i)
Base.getindex(d::AbstractGrid, i::Tuple) = getindex(dims(d), i)
Base.ndims(d::AbstractGrid) = length(dims(d))
Base.size(d::AbstractGrid) = map(length, dims(d))
Base.length(d::AbstractGrid) = prod(size(d))
Base.firstindex(d::AbstractGrid) = 1
Base.lastindex(d::AbstractGrid) = length(d)
Base.axes(d::AbstractGrid) = axes(dims(d))
Base.iterate(d::AbstractGrid, i::Int = 1) = iterate(dims(d), i)
Base.map(f, d::AbstractGrid) = rebuild(typeof(d), map(f, dims(d)), header(d))
#Make sure we actually get a tuple here
Base.map(f, args, d::AbstractGrid) = map(f, args, dims(d))
Base.map(f, d::AbstractGrid, args) = map(f, dims(d), args)
Base.front(d::AbstractGrid) = Base.front(dims(d))
Base.eltype(d::AbstractGrid) = Base.eltype(dims(d))

function Base.show(io::IO, mime::MIME"text/plain", x::RectiGrid)
    println(io, "RectiGrid:")
    for n in propertynames(x)
        print(io, "\t")
        show(io, mime, getproperty(x, n))
        println(io)
    end
end


Base.propertynames(d::RectiGrid) = keys(d)
Base.getproperty(g::RectiGrid, p::Symbol) = basedim(dims(g)[findfirst(==(p), keys(g))])

"""
    RectiGrid{Na}(dims::Tuple, header=ComradeBase.NoHeader) where {Na}

Builds the EHT image dimensions using the names `Na` and dimensions `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.
The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :Ti, :F)
    - (:X, :Y, :F,  :Ti)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
# Notes
Instead use the direct [`IntensityMap`](@ref) function.
```julia
dims = RectiGrid((X=-5.0:0.1:5.0, Y=-4.0:0.1:4.0, Ti=[1.0, 1.5, 1.75], F=[230, 345]))
```
"""
@inline function RectiGrid(dims::Tuple, header::AbstractHeader=NoHeader())
    return RectiGrid{typeof(dims), typeof(header)}(dims, header)
end


# This is needed to prevent doubling up on the dimension
@inline function RectiGrid(dims::NamedTuple{Na, T}, header::AbstractHeader=NoHeader()) where {Na, N, T<:NTuple{N, DD.Dimension}}
    return RectiGrid(values(dims), header)
end

"""
    RectiGrid(dims::NamedTuple{Na}, header=ComradeBase.NoHeader())

Builds the EHT image dimensions using the names `Na` and dimensions are the values of `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.
The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :Ti, :F)
    - (:X, :Y, :F, :Ti)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:Ti` is the
the time direction and `:F` is the frequency direction.
# Notes
Instead use the direct [`IntensityMap`](@ref) function.
```julia
dims = RectiGrid((X=-5.0:0.1:5.0, Y=-4.0:0.1:4.0, Ti=[1.0, 1.5, 1.75], Fr=[230, 345]))
```
"""
@inline function RectiGrid(nt::NamedTuple, header::AbstractHeader=ComradeBase.NoHeader())
    dims = map(keys(nt), values(nt)) do k,v
        DD.rebuild(DD.key2dim(k), v)
    end
    return RectiGrid(format(dims, map(eachindex, dims)), header)
end

function DD.rebuild(::Type{<:RectiGrid}, g, header=ComradeBase.NoHeader())
    RectiGrid(g, header)
end


# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)},
                        <:NamedTuple{(:X, :Y, :T)}, <:NamedTuple{(:X, :Y, :F)},
                        <:NamedTuple{(:X,:Y)}}
# const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}


# # Our image will be some KeyedArray but where we require specific keys names
