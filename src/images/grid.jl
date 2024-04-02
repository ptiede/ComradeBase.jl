# In this file we will define our base image class. This is entirely based on
export RectiGrid, named_dims, dims, header, axisdims, executor

abstract type AbstractGrid{D, E} end


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


abstract type AbstractRectiGrid{D, E} <: AbstractGrid{D, E} end

struct Serial end

executor(g::AbstractGrid) = getfield(g, :executor)


"""
    RectiGrid(dims::Tuple, header=ComradeBase.NoHeader)

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

# Notes
Warning it is rare you need to access this constructor directly. For spatial intensitymaps
just use the [`imagepixels`](@ref) function.
"""
struct RectiGrid{D, E, Hd<:AbstractHeader} <: AbstractRectiGrid{D, E}
    dims::D
    executor::E
    header::Hd
    @inline function RectiGrid(dims::Tuple; executor=Serial(), header::AbstractHeader=NoHeader())
        df = _format_dims(dims)
        return new{typeof(df), typeof(executor), typeof(header)}(df, executor, header)
    end

end


function _format_dims(dg::Tuple)
    return DD.format(dg, map(eachindex, dg))
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
Base.keys(g::AbstractGrid) = throw(MethodError(Base.keys, "You must implement `Base.keys($(typeof(g)))`"))

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

function Base.show(io::IO, mime::MIME"text/plain", x::RectiGrid{D, E}) where {D, E}
    println(io, "RectiGrid(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    for n in propertynames(x)
        print(io, "\t")
        print(io, n, ": ")
        show(io, mime, getproperty(x, n))
        println(io)
    end
    print(io, ")")
end


Base.propertynames(d::RectiGrid) = keys(d)
Base.getproperty(g::RectiGrid, p::Symbol) = basedim(dims(g)[findfirst(==(p), keys(g))])



# This is needed to prevent doubling up on the dimension
@inline function RectiGrid(dims::NamedTuple{Na, T}; executor=Serial(), header::AbstractHeader=NoHeader()) where {Na, N, T<:NTuple{N, DD.Dimension}}
    return RectiGrid(values(dims), executor, header)
end

@noinline function _make_dims(ks, vs)
    ds = DD.key2dim(ks)
    return map(ds, vs) do d,v
        DD.rebuild(d, v)
    end
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
@inline function RectiGrid(nt::NamedTuple, executor=Serial(), header::AbstractHeader=ComradeBase.NoHeader())
    dims = _make_dims(keys(nt), values(nt))
    return RectiGrid(dims; executor, header)
end

function DD.rebuild(::Type{<:RectiGrid}, g, executor=Serial(), header=ComradeBase.NoHeader())
    RectiGrid(g; executor, header)
end


# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)},
                        <:NamedTuple{(:X, :Y, :T)}, <:NamedTuple{(:X, :Y, :F)},
                        <:NamedTuple{(:X,:Y)}}
# const DataArr = Union{NDA{(:X, :Y, :T, :F)}, NDA{(:X, :Y, :F, :T)}, NDA{(:X, :Y)}}



# Now we have a completely unstructured grid or really a vector of points
struct UnstructuredGrid{D, E, H<:AbstractHeader} <: AbstractGrid{D,E}
    dims::D
    executor::E
    header::H
end

function UnstructuredGrid(nt::NamedTuple; executor=Serial(), header=NoHeader())
    return UnstructuredGrid(StructArray(nt), executor, header)
end

Base.ndims(d::UnstructuredGrid) = ndims(dims(d))
Base.size(d::UnstructuredGrid) = size(dims(d))
Base.firstindex(d::UnstructuredGrid) = firstindex(dims(d))
Base.lastindex(d::UnstructuredGrid) = lastindex(dims(d))
#Make sure we actually get a tuple here
Base.front(d::UnstructuredGrid) = Base.front(dims(d))
Base.eltype(d::UnstructuredGrid) = Base.eltype(dims(d))

function DD.rebuild(::Type{<:UnstructuredGrid}, g, executor=Serial(), header=ComradeBase.NoHeader())
    UnstructuredGrid(g, executor, header)
end

Base.propertynames(g::UnstructuredGrid) = propertynames(dims(g))
Base.getproperty(g::UnstructuredGrid, p::Symbol) = getproperty(dims(g), p)
