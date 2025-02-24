# In this file we will define our base image class. This is entirely based on
export RectiGrid, UnstructuredDomain, domainpoints,
       named_dims, dims, header, axisdims, executor,
       posang, update_spat, rotmat

abstract type AbstractDomain end
abstract type AbstractSingleDomain{D,E} <: AbstractDomain end

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
    return allocate_vismap(ispolarized(typeof(m)), m, g)
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
    return allocate_imgmap(ispolarized(typeof(m)), m, g)
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
# ChainRulesCore.@non_differentiable executor(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(executor), args...) = nothing

"""
    dims(g::AbstractSingleDomain)

Returns a tuple containing the dimensions of `g`. For a named version see [`ComradeBase.named_dims`](@ref)
"""
DD.dims(g::AbstractSingleDomain) = getfield(g, :dims)
# ChainRulesCore.@non_differentiable DD.dims(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(DD.dims), x::AbstractSingleDomain) = nothing

"""
    named_dims(g::AbstractSingleDomain)

Returns a named tuple containing the dimensions of `g`. For a unnamed version see [`dims`](@ref)
"""
named_dims(g::AbstractSingleDomain) = NamedTuple{keys(g)}(dims(g))
# ChainRulesCore.@non_differentiable named_dims(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(named_dims), args...) = nothing

"""
    header(g::AbstractSingleDomain)

Returns the headerinformation of the dimensions `g`
"""
header(g::AbstractSingleDomain) = getfield(g, :header)
# ChainRulesCore.@non_differentiable header(::AbstractSingleDomain)
EnzymeRules.inactive(::typeof(header), args...) = nothing
function Base.keys(g::AbstractSingleDomain)
    throw(MethodError(Base.keys, "You must implement `Base.keys($(typeof(g)))`"))
end

# We index the dimensions not the grid itself
Base.getindex(d::AbstractSingleDomain, i::Int) = getindex(dims(d), i)

Base.ndims(d::AbstractSingleDomain) = length(dims(d))
Base.size(d::AbstractSingleDomain) = map(length, dims(d))
Base.length(d::AbstractSingleDomain) = prod(size(d))
Base.firstindex(d::AbstractSingleDomain) = 1
Base.lastindex(d::AbstractSingleDomain) = length(d)
Base.axes(d::AbstractSingleDomain) = axes(dims(d))
Base.iterate(d::AbstractSingleDomain, i::Int=1) = iterate(dims(d), i)
# Base.front(d::AbstractSingleDomain) = Base.front(dims(d))
# We return the eltype of the dimensions. Should we change this?
Base.eltype(d::AbstractSingleDomain) = eltype(basedim(first(dims(d))))

# These aren't needed and I am not sure if the semantics are what we actually want
# Base.map(f, d::AbstractSingleDomain) = rebuild(typeof(d), map(f, dims(d)), executor(d), header(d))
# Base.map(f, args, d::AbstractSingleDomain) = map(f, args, dims(d))
# Base.map(f, d::AbstractSingleDomain, args) = rebuild(typeof(d), map(f, dims(d), args))

const AMeta = DimensionalData.Dimensions.Lookups.AbstractMetadata

abstract type AbstractHeader{T,X} <: AMeta{T,X} end

"""
    MinimalHeader{T}

A minimal header type for ancillary image information.

# Fields
$(FIELDS)
"""
struct MinimalHeader{T} <: AbstractHeader{T,NamedTuple{(),Tuple{}}}
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
const NoHeader = DimensionalData.NoMetadata

abstract type AbstractRectiGrid{D,E} <: AbstractSingleDomain{D,E} end
create_map(array, g::AbstractRectiGrid) = IntensityMap(array, g)
function allocate_map(M::Type{<:AbstractArray{T}}, g::AbstractRectiGrid) where {T}
    return IntensityMap(similar(M, size(g)), g)
end

function fieldofview(dims::AbstractRectiGrid)
    (; X, Y) = dims
    dx = step(X)
    dy = step(Y)
    return (X=abs(last(X) - first(X)) + dx, Y=abs(last(Y) - first(Y)) + dy)
end

@inline posang(d::AbstractRectiGrid) = getfield(d, :posang)

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

struct RectiGrid{D,E,Hd<:AMeta,P} <: AbstractRectiGrid{D,E}
    dims::D
    executor::E
    header::Hd
    posang::P
    @inline function RectiGrid(dims::Tuple; executor=Serial(),
                               header::AMeta=NoHeader(), posang=zero(eltype(first(dims))))
        df = DD.format(dims)
        return new{typeof(df),typeof(executor),typeof(header),typeof(posang)}(df, executor,
                                                                              header,
                                                                              posang)
    end
end

EnzymeRules.inactive_type(::Type{<:RectiGrid}) = true

function rotmat(d::RectiGrid)
    s, c = sincos(posang(d))
    r = (c, s, -s, c)
    m = SMatrix{2,2,typeof(s),4}(r)
    return m
end

function domainpoints(d::RectiGrid{D,Hd}) where {D,Hd}
    g = map(basedim, dims(d))
    rot = rotmat(d)
    N = keys(d)
    return RotGrid(StructArray(NamedTuple{N}(_build_slices(g, size(d)))), rot)
end

struct RotGrid{T,N,G<:AbstractArray{T,N},M} <: AbstractArray{T,N}
    grid::G
    rot::M
end

@inline function update_spat(p::NamedTuple, x, y)
    p1 = @set p[1] = x
    p2 = @set p1[2] = y
    return p2
end

Base.parent(g::RotGrid) = getfield(g, :grid)
Base.getproperty(g::RotGrid, p::Symbol) = getproperty(parent(g), p)
Base.propertynames(g::RotGrid) = propertynames(parent(g))
Base.size(g::RotGrid) = size(parent(g))
Base.IndexStyle(::Type{<:RotGrid{T,N,G}}) where {T,N,G} = Base.IndexStyle(G)
Base.firstindex(g::RotGrid) = firstindex(parent(g))
Base.lastindex(g::RotGrid) = lastindex(parent(g))
Base.axes(g::RotGrid) = axes(parent(g))
rotmat(g::RotGrid) = getfield(g, :rot)

Base.@propagate_inbounds function Base.getindex(g::RotGrid, i::Int)
    p = getindex(parent(g), i)
    pr = rotmat(g) * SVector(values(p)[1:2])
    return update_spat(p, pr[1], pr[2])
end

Base.@propagate_inbounds function Base.getindex(g::RotGrid, I::Vararg{Int})
    p = getindex(parent(g), I...)
    pr = rotmat(g) * SVector(values(p)[1:2])
    return update_spat(p, pr[1], pr[2])
end

# Use structarray broadcasting
Base.BroadcastStyle(::Type{<:RotGrid{T,N,G}}) where {T,N,G} = Base.BroadcastStyle(G)

Base.keys(g::RectiGrid) = map(name, dims(g))

@inline RectiGrid(g::RectiGrid) = g

function Base.show(io::IO, mime::MIME"text/plain", x::RectiGrid{D,E}) where {D,E}
    println(io, "RectiGrid(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    return print(io, "\n)")
end

Base.propertynames(d::RectiGrid) = keys(d)
Base.getproperty(g::RectiGrid, p::Symbol) = basedim(dims(g)[findfirst(==(p), keys(g))])

# This is needed to prevent doubling up on the dimension
@inline function RectiGrid(dims::NamedTuple{Na,T}; executor=Serial(),
                           header::AMeta=NoHeader(),
                           posang=zero(eltype(first(values(dims))))) where {Na,N,
                                                                            T<:NTuple{N,
                                                                                      DD.Dimension}}
    return RectiGrid(values(dims); executor, header, posang)
end

@noinline function _make_dims(ks, vs)
    ds = DD.name2dim(ks)
    return map(ds, vs) do d, v
        return DD.rebuild(d, v)
    end
end

"""
    RectiGrid(dims::NamedTuple{Na}; executor=Serial(), header=ComradeBase.NoHeader(), posang=0.0)
    RectiGrid(dims::NTuple{N, <:DimensionalData.Dimension}; executor=Serial(), header=ComradeBase.NoHeader(), posang=0.0)

Creates a rectilinear grid of pixels with the dimensions `dims`. The convention is that
the first two dimension are the spatial dimensions `X/U` and `Y/V`. The remaining dimensions
can be anything, for example:
  - (:X, :Y, :Ti, :Fr)
  - (:X, :Y, :Fr, :Ti)
  - (:X, :Y) # spatial only

where `X/U,Y/V` are the RA and DEC spatial dimensions in image/visibility space respectively, 
`Ti` is the time dimension and `Fr` is the frequency dimension.

Note that the majority of the time users should just call [`imagepixels`](@ref) to create
a spatial grid.

## Optional Arguments

 - `executor`: specifies how different models
    are executed. The default is `Serial` which mean serial CPU computations. For threaded
    computations use [`ThreadsEx()`](@ref) or load `OhMyThreads.jl` to uses their schedulers.
 - `header`: specified underlying header information for the grid. This is used to store
    information about the image such as the source, RA and DEC, MJD.
 - `posang`: specifies the position angle of the grid, relative to RA=0 axis. Note that when 
             `posang != 0` the X and Y coordinate are relative to the rotated grid and not
             the on sky RA and DEC orientation. To see the true on sky points you can access
             them by calling `domainpoints(grid)`.

## Examples

```julia
dims = RectiGrid((X(-5.0:0.1:5.0), Y(-4.0:0.1:4.0), Ti([1.0, 1.5, 1.75]), Fr([230, 345])); executor=ThreadsEx())
dims = RectiGrid((X = -5.0:0.1:5.0, Y = -4.0:0.1:4.0, Ti = [1.0, 1.5, 1.75], Fr = [230, 345]); executor=ThreadsEx()))
```
"""
@inline function RectiGrid(nt::NamedTuple; executor=Serial(),
                           header::AMeta=ComradeBase.NoHeader(),
                           posang=zero(eltype(first(values(nt)))))
    dims = _make_dims(keys(nt), values(nt))
    return RectiGrid(dims; executor, header, posang)
end

function DD.rebuild(grid::RectiGrid, dims, executor=executor(grid),
                    header=metadata(grid), posang=posang(grid))
    return RectiGrid(dims; executor, header, posang)
end

# Define some helpful names for ease typing
