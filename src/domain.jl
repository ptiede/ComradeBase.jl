# In this file we will define our base image class. This is entirely based on
export RectiGrid, UnstructuredDomain, domainpoints,
       named_dims, dims, header, axisdims, executor,
       Serial, ThreadsEx


abstract type AbstractDomain end
abstract type AbstractSingleDomain{D, E} <: AbstractDomain end

"""
    create_map(array, g::AbstractSingleDomain)

Create a map of values specialized by the grid `g`.
"""
function create_map end

"""
    create_vismap(array, g::AbstractSingleDomain)

Create a map of values specialized by the grid `g` in the visibility domain.
The default is to call `create_map` with the same arguments.
"""
create_vismap(array, g::AbstractDomain) = create_map(array, g)

"""
    create_imgmap(array, g::AbstractSingleDomain)

Create a map of values specialized by the grid `g` in the image domain.
The default is to call `create_map` with the same arguments.
"""
create_imgmap(array, g::AbstractDomain) = create_map(array, g)

"""
    allocate_imgmap(m::AbstractModel, g::AbstractSingleDomain)

Allocate the default map specialized by the grid `g`
"""
function allocate_imgmap end

"""
    allocate_vismap(m::AbstractModel, g::AbstractSingleDomain)

Allocate the default map specialized by the grid `g`
"""
function allocate_vismap end

function allocate_vismap(m::AbstractModel, g::AbstractDomain)
    allocate_vismap(ispolarized(typeof(m)), m, g)
end

function allocate_vismap(::IsPolarized, m::AbstractModel, g::AbstractDomain)
    M = StructArray{StokesParams{Complex{eltype(g)}}}
    return allocate_map(M, g)
end

function allocate_vismap(::NotPolarized, m::AbstractModel, g::AbstractDomain)
    M = Array{Complex{eltype(g)}}
    return allocate_map(M, g)
end

function allocate_imgmap(m::AbstractModel, g::AbstractDomain)
    allocate_imgmap(ispolarized(typeof(m)), m, g)
end

function allocate_imgmap(::IsPolarized, m::AbstractModel, g::AbstractDomain)
    M = StructArray{StokesParams{eltype(g)}}
    return allocate_map(M, g)
end

function allocate_imgmap(::NotPolarized, m::AbstractModel, g::AbstractDomain)
    M = Array{eltype(g)}
    return allocate_map(M, g)
end





"""
    domainpoints(g::AbstractSingleDomain)

Create a grid iterator that can be used to iterate through different points.
All grid methods must implement this method.
"""
function domainpoints end

# We enforce that all grids are static for performance reasons
# If this is not true please create a custom subtype
# ChainRulesCore.@non_differentiable domainpoints(d::AbstractSingleDomain)
# EnzymeRules.inactive(::typeof(domainpoints), args...) = nothing


"""
    executor(g::AbstractSingleDomain)

Returns the executor used to compute the intensitymap or visibilitymap
"""
executor(g::AbstractSingleDomain) = getfield(g, :executor)
ChainRulesCore.@non_differentiable executor(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(executor), args...) = nothing


"""
    dims(g::AbstractSingleDomain)

Returns a tuple containing the dimensions of `g`. For a named version see [`ComradeBase.named_dims`](@ref)
"""
DD.dims(g::AbstractSingleDomain) = getfield(g, :dims)
ChainRulesCore.@non_differentiable DD.dims(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(DD.dims), x::AbstractSingleDomain) = nothing


"""
    named_dims(g::AbstractSingleDomain)

Returns a named tuple containing the dimensions of `g`. For a unnamed version see [`dims`](@ref)
"""
named_dims(g::AbstractSingleDomain) = NamedTuple{keys(g)}(dims(g))
ChainRulesCore.@non_differentiable named_dims(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(named_dims), args...) = nothing


"""
    header(g::AbstractSingleDomain)

Returns the headerinformation of the dimensions `g`
"""
header(g::AbstractSingleDomain) = getfield(g, :header)
ChainRulesCore.@non_differentiable header(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(header), args...) = nothing
Base.keys(g::AbstractSingleDomain) = throw(MethodError(Base.keys, "You must implement `Base.keys($(typeof(g)))`"))

"""
    Serial()

Uses serial execution when computing the intensitymap or visibilitymap
"""
struct Serial end

"""
    ThreadsEx(;scheduler::Symbol = :dynamic)

Uses Julia's Threads @threads macro when computing the intensitymap or visibilitymap.
You can choose from Julia's various schedulers by passing the scheduler as a parameter.
The default is :dynamic, but it isn't considered part of the stable API and may change
at any moment.
"""
struct ThreadsEx{S} end
ThreadsEx() = ThreadsEx(:dynamic)
ThreadsEx(s) = ThreadsEx{s}()

#TODO can this be made nicer?
@static if VERSION ≥ v"1.11"
    const schedulers = (:(:dynamic), :(:static), :(:greedy))
else
    const schedulers = (:(:dynamic), :(:static))
end


# We index the dimensions not the grid itself
Base.getindex(d::AbstractSingleDomain, i::Int) = getindex(dims(d), i)


Base.ndims(d::AbstractSingleDomain) = length(dims(d))
Base.size(d::AbstractSingleDomain) = map(length, dims(d))
Base.length(d::AbstractSingleDomain) = prod(size(d))
Base.firstindex(d::AbstractSingleDomain) = 1
Base.lastindex(d::AbstractSingleDomain) = length(d)
Base.axes(d::AbstractSingleDomain) = axes(dims(d))
Base.iterate(d::AbstractSingleDomain, i::Int = 1) = iterate(dims(d), i)
# Base.front(d::AbstractSingleDomain) = Base.front(dims(d))
# We return the eltype of the dimensions. Should we change this?
Base.eltype(d::AbstractSingleDomain) = eltype(basedim(first(dims(d))))

# These aren't needed and I am not sure if the semantics are what we actually want
# Base.map(f, d::AbstractSingleDomain) = rebuild(typeof(d), map(f, dims(d)), executor(d), header(d))
# Base.map(f, args, d::AbstractSingleDomain) = map(f, args, dims(d))
# Base.map(f, d::AbstractSingleDomain, args) = rebuild(typeof(d), map(f, dims(d), args))


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


abstract type AbstractRectiGrid{D, E} <: AbstractSingleDomain{D, E} end
create_map(array, g::AbstractRectiGrid) = IntensityMap(array, g)
allocate_map(M::Type{<:AbstractArray}, g::AbstractRectiGrid) = IntensityMap(similar(M, size(g)), g)

function fieldofview(dims::AbstractRectiGrid)
    (;X,Y) = dims
    dx = step(X)
    dy = step(Y)
    (X=abs(last(X) - first(X))+dx, Y=abs(last(Y)-first(Y))+dy)
end


"""
    pixelsizes(img::IntensityMap)
    pixelsizes(img::AbstractRectiGrid)

Returns a named tuple with the spatial pixel sizes of the image.
"""
function pixelsizes(keys::AbstractRectiGrid)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end



struct RectiGrid{D, E, Hd<:AbstractHeader} <: AbstractRectiGrid{D, E}
    dims::D
    executor::E
    header::Hd
    @inline function RectiGrid(dims::Tuple; executor=Serial(), header::AbstractHeader=NoHeader())
        df = _format_dims(dims)
        return new{typeof(df), typeof(executor), typeof(header)}(df, executor, header)
    end

end

function domainpoints(d::RectiGrid{D, Hd}) where {D, Hd}
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
    return RectiGrid(values(dims); executor, header)
end

@noinline function _make_dims(ks, vs)
    ds = DD.name2dim(ks)
    return map(ds, vs) do d,v
        DD.rebuild(d, v)
    end
end


"""
    RectiGrid(dims::NamedTuple{Na}; executor=Serial(), header=ComradeBase.NoHeader())
    RectiGrid(dims::NTuple{N, <:DimensionalData.Dimension}; executor=Serial(), header=ComradeBase.NoHeader())

Creates a rectilinear grid of pixels with the dimensions `dims`. The dims can either be
a named tuple of dimensions or a tuple of dimensions. The dimensions can be in any order
however the standard orders are:
  - (:X, :Y, :Ti, :Fr)
  - (:X, :Y, :Fr, :Ti)
  - (:X, :Y) # spatial only

where `X,Y` are the RA and DEC spatial dimensions respectively, `Ti` is the time dimension
and `Fr` is the frequency dimension.

Note that the majority of the time users should just call [`imagepixels`](@ref) to create
a spatial grid.

## Optional Arguments

 - `executor`: specifies how different models
    are executed. The default is `Serial` which mean serial CPU computations. For threaded
    computations use [`ThreadsEx()`](@ref) or load `OhMyThreads.jl` to uses their schedulers.
 - `header`: specified underlying header information for the grid. This is used to store
    information about the image such as the source, RA and DEC, MJD.

## Examples

```julia
dims = RectiGrid((X(-5.0:0.1:5.0), Y(-4.0:0.1:4.0), Ti([1.0, 1.5, 1.75]), Fr([230, 345])); executor=ThreadsEx())
dims = RectiGrid((X = -5.0:0.1:5.0, Y = -4.0:0.1:4.0, Ti = [1.0, 1.5, 1.75], Fr = [230, 345]); executor=ThreadsEx()))
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



# TODO make this play nice with dimensional data
struct UnstructuredDomain{D,E, H<:AbstractHeader} <: AbstractSingleDomain{D,E}
    dims::D
    executor::E
    header::H
end

"""
    UnstructuredDomain(dims::NamedTuple; executor=Serial(), header=ComradeBase.NoHeader)

Builds an unstructured grid (really a vector of points) from the dimensions `dims`.
The `executor` is used controls how the grid is computed when calling
`visibilitymap` or `intensitymap`. The default is `Serial` which mean regular CPU computations.
For threaded execution use [`ThreadsEx()`](@ref) or load `OhMyThreads.jl` to uses their schedulers.

Note that unlike `RectiGrid` which assigns dimensions to the grid points, `UnstructuredDomain`
does not. This is becuase the grid is unstructured the points are a cloud in a space
"""
function UnstructuredDomain(nt::NamedTuple; executor=Serial(), header=NoHeader())
    p = StructArray(nt)
    return UnstructuredDomain(p, executor, header)
end

Base.ndims(d::UnstructuredDomain) = ndims(dims(d))
Base.size(d::UnstructuredDomain) = size(dims(d))
Base.firstindex(d::UnstructuredDomain) = firstindex(dims(d))
Base.lastindex(d::UnstructuredDomain) = lastindex(dims(d))
#Make sure we actually get a tuple here
# Base.front(d::UnstructuredDomain) = UnstructuredDomain(Base.front(StructArrays.components(dims(d))), executor=executor(d), header=header(d))
# Base.eltype(d::UnstructuredDomain) = Base.eltype(dims(d))

function DD.rebuild(::Type{<:UnstructuredDomain}, g, executor=Serial(), header=ComradeBase.NoHeader())
    UnstructuredDomain(g, executor, header)
end

Base.propertynames(g::UnstructuredDomain) = propertynames(domainpoints(g))
Base.getproperty(g::UnstructuredDomain, p::Symbol) = getproperty(domainpoints(g), p)
Base.keys(g::UnstructuredDomain) = propertynames(g)
named_dims(g::UnstructuredDomain) = StructArrays.components(dims(g))

function domainpoints(d::UnstructuredDomain)
    return getfield(d, :dims)
end


function Base.summary(io::IO, g::UnstructuredDomain)
    n = propertynames(domainpoints(g))
    printstyled(io, "│ "; color=:light_black)
    print(io, "UnstructuredDomain with dims: $n")
end

function Base.show(io::IO, mime::MIME"text/plain", x::UnstructuredDomain)
    println(io, "UnstructredDomain(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    print(io, "\n)")
end


create_map(array, g::UnstructuredDomain) = UnstructuredMap(array, g)
allocate_map(M::Type{<:A}, g::UnstructuredDomain) where {A<:AbstractArray} = UnstructuredMap(similar(M, size(g)), g)
