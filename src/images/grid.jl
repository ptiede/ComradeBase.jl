# In this file we will define our base image class. This is entirely based on
export RectiGrid, UnstructuredGrid,
       named_dims, dims, header, axisdims, executor,
       Serial, ThreadsEx

abstract type AbstractGrid{D, E} end

"""
    executor(g::AbstractGrid)

Returns the executor used to compute the intensitymap or visibilitymap
"""
executor(g::AbstractGrid) = getfield(g, :executor)
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

"""
    Serial()

Uses serial execution when computing the intensitymap or visibilitymap
"""
struct Serial end

"""
    ThreadsEx(scheduler::Symbol = :dynamic)

Uses Julia's Threads @threads macro when computing the intensitymap or visibilitymap.
You can choose from Julia's various schedulers by passing the scheduler as a parameter.
The default is :dynamic, but it isn't considered part of the stable API and may change
at any moment.
"""
struct ThreadsEx{S} end
ThreadsEx() = ThreadsEx(:dynamic)
ThreadsEx(s) = ThreadsEx{s}()


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
Base.eltype(d::AbstractGrid) = Base.eltype(Base.eltype(dims(d)))



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




"""
    RectiGrid(dims::Tuple, header=ComradeBase.NoHeader)

Builds the EHT image dimensions using the names `Na` and dimensions `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.
The type parameter `Na` defines the names of each dimension.
For image domain grids the names are usually one of
  - (:X, :Y, :Ti, :Fr)
  - (:X, :Y, :Fr,  :Ti)
  - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:Ti` is the
the time direction and `:Fr` is the frequency direction. For visibility domain
the dimensions usually are:
  - (:U, :V, :Ti, :Fr)
  - (:U, :V, :Fr, :Fr)
  - (:U, :V) # spatial only
# Notes
Instead use the direct [`IntensityMap`](@ref) function.
```julia
dims = RectiGrid((X=-5.0:0.1:5.0, Y=-4.0:0.1:4.0, Ti=[1.0, 1.5, 1.75], Fr=[230, 345]))
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

function imagegrid(d::RectiGrid{D, Hd}) where {D, Hd}
    g = map(basedim, dims(d))
    N = keys(d)
    return StructArray(NamedTuple{N}(_build_slices(g, size(d))))
end


function _format_dims(dg::Tuple)
    return DD.format(dg, map(eachindex, dg))
end

Base.keys(g::RectiGrid) = map(name, dims(g))

@inline RectiGrid(g::RectiGrid) = g


function Base.show(io::IO, mime::MIME"text/plain", x::RectiGrid{D, E}) where {D, E}
    println(io, "RectiGrid(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    print(io, "\n)")
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
@inline function RectiGrid(nt::NamedTuple; executor=Serial(), header::AbstractHeader=ComradeBase.NoHeader())
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



# TODO make this play nice with dimensional data
struct UnstructuredGrid{D,E, H<:AbstractHeader} <: AbstractGrid{D,E}
    dims::D
    executor::E
    header::H
end

"""
    UnstructuredGrid(dims::AbstractArray; executor=Serial(), header=ComradeBase.NoHeader)

Builds an unstructured grid (really a vector of points) from the dimensions `dims`.
The `executor` is used controls how the grid is computed when calling
`visibilitymap` or `intensitymap`.

Note that unlike `RectiGrid` which assigns dimensions to the grid points, `UnstructuredGrid`
does not. This is becuase the grid is unstructured the points are a cloud in a space
"""
function UnstructuredGrid(nt::NamedTuple; executor=Serial(), header=NoHeader())
    p = StructArray(nt)
    pnt = (position=p,)
    dims = _make_dims(keys(pnt), values(pnt))
    # df = _format_dims(dims)
    return UnstructuredGrid(dims, executor, header)
end

Base.ndims(d::UnstructuredGrid) = ndims(dims(d))
Base.size(d::UnstructuredGrid) = size(dims(d))
Base.firstindex(d::UnstructuredGrid) = firstindex(dims(d))
Base.lastindex(d::UnstructuredGrid) = lastindex(dims(d))
#Make sure we actually get a tuple here
Base.front(d::UnstructuredGrid) = Base.front(dims(d))
Base.eltype(d::UnstructuredGrid) = Base.eltype(dims(d))

function DD.rebuild(::Type{<:UnstructuredGrid}, g::Tuple, executor=Serial(), header=ComradeBase.NoHeader())
    UnstructuredGrid(g, executor, header)
end

Base.propertynames(g::UnstructuredGrid) = propertynames(imagegrid(g))
Base.getproperty(g::UnstructuredGrid, p::Symbol) = getproperty(imagegrid(g), p)

function imagegrid(d::UnstructuredGrid)
    return basedim(dims(d)[1])
end


function Base.summary(io::IO, g::UnstructuredGrid)
    n = propertynames(imagegrid(g))
    printstyled(io, "│ "; color=:light_black)
    print(io, "UnstructuredGrid with dims: $n")

end
