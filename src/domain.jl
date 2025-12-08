# In this file we will define our base image class. This is entirely based on
export domainpoints,
    named_dims, dims, header, axisdims, executor,
    posang, update_spat, rotmat

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

function allocate_vismap(m::AbstractModel, g::AbstractSingleDomain{D, E}) where {D, E}
    return allocate_vismap(ispolarized(typeof(m)), m, g)
end

function allocate_imgmap(m::AbstractModel, g::AbstractSingleDomain)
    return allocate_imgmap(ispolarized(typeof(m)), m, g)
end

@inline function similartype(::IsPolarized, E, T)
    return StructArray{StokesParams{T}}
end

@inline function similartype(::NotPolarized, E, T)
    return Array{T}
end

function allocate_vismap(p, m::AbstractModel, g::AbstractSingleDomain{D, E}) where {D, E}
    M = similartype(p, E, Complex{eltype(g)})
    return allocate_map(M, g)
end


function allocate_imgmap(p, ::AbstractModel, g::AbstractSingleDomain{D, E}) where {D, E}
    M = similartype(p, E, eltype(g))
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
EnzymeRules.inactive(::typeof(domainpoints), args...) = nothing

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
Base.iterate(d::AbstractSingleDomain, i::Int = 1) = iterate(dims(d), i)
# Base.front(d::AbstractSingleDomain) = Base.front(dims(d))
# We return the eltype of the dimensions. Should we change this?
Base.eltype(d::AbstractSingleDomain) = eltype(basedim(first(dims(d))))

const AMeta = DimensionalData.Dimensions.Lookups.AbstractMetadata

abstract type AbstractHeader{T, X} <: AMeta{T, X} end

"""
    MinimalHeader{T}

A minimal header type for ancillary image information.

# Fields
$(FIELDS)
"""
struct MinimalHeader{T} <: AbstractHeader{T, NamedTuple{(), Tuple{}}}
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

function DimensionalData.val(m::AbstractHeader)
    n = propertynames(m)
    pm = Base.Fix1(getproperty, m)
    return NamedTuple{n}(map(pm, n))
end

"""
    NoHeader


"""
const NoHeader = DimensionalData.NoMetadata

abstract type AbstractRectiGrid{D, E} <: AbstractSingleDomain{D, E} end
create_map(array, g::AbstractRectiGrid) = IntensityMap(array, g)
function allocate_map(M::Type{<:AbstractArray{T}}, g::AbstractRectiGrid) where {T}
    return IntensityMap(similar(M, size(g)), g)
end

function fieldofview(dims::AbstractRectiGrid)
    (; X, Y) = dims
    dx = step(X)
    dy = step(Y)
    return (X = abs(last(X) - first(X)) + dx, Y = abs(last(Y) - first(Y)) + dy)
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
    return (X = step(x), Y = step(y))
end

# Define some helpful names for ease typing
