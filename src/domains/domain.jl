# In this file we will define our base image class. This is entirely based on
export domainpoints,
    named_dims, dims, header, axisdims, executor,
    posang, update_spat, rotmat, imgdomain, visdomain

abstract type AbstractDomain end
abstract type AbstractSingleDomain{D, E} <: AbstractDomain end

"""
    AbstractDualDomain

Defines a domain that is a combination of a image and visibility domain.
There are two methods that must be implemented for this domain:
    - `imgdomain(d::AbstractDualDomain)` which returns the image domain
    - `visdomain(d::AbstractDualDomain)` which returns the visibility domain
"""
abstract type AbstractDualDomain <: AbstractDomain end
imgdomain(d::AbstractDualDomain) = getfield(d, :imgdomain)
visdomain(d::AbstractDualDomain) = getfield(d, :visdomain)


"""
    create_map(array, g::AbstractSingleDomain)

Create a map of values specialized by the grid `g`.
"""
create_map(array, g::AbstractSingleDomain) = IntensityMap(array, g)

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

function allocate_vismap(m::AbstractModel, g::AbstractSingleDomain)
    return allocate_vismap(ispolarized(typeof(m)), m, g)
end

function allocate_imgmap(m::AbstractModel, g::AbstractSingleDomain)
    return allocate_imgmap(ispolarized(typeof(m)), m, g)
end

"""
    _storage(executor, T, sz)

Allocates uninitialized map storage with element type `T` and size `sz` for `executor`.
Executor extensions add methods for their executor type.
"""
_storage(executor, ::Type{T}, sz) where {T} = Array{T}(undef, sz)

"""
    allocate_map(M::Type{<:AbstractArray}, g::AbstractSingleDomain, eldims::Tuple = ())

Allocates an uninitialized `IntensityMap` over `g` whose storage is an array of type `M` with
the dims of `g` followed by the dims `eldims`, e.g. `(Stokes(...),)` for a [`StokesMap`](@ref).
"""
allocate_map(M, g) = allocate_map(M, g, ())
function allocate_map(M::Type{<:AbstractArray}, g::AbstractSingleDomain, eldims::Tuple)
    storage = similar(M, (size(g)..., map(length, eldims)...))
    return _wrapstorage(storage, g, (), Symbol(""), eldims)
end

function _allocate_map(p, ::Type{T}, g::AbstractSingleDomain) where {T}
    eldims = _polarizationdims(p)
    storage = _storage(executor(g), T, (size(g)..., map(length, eldims)...))
    return _wrapstorage(storage, g, (), Symbol(""), eldims)
end

allocate_vismap(p, ::AbstractModel, g::AbstractSingleDomain) = _allocate_map(p, complex(eltype(g)), g)
allocate_imgmap(p, ::AbstractModel, g::AbstractSingleDomain) = _allocate_map(p, eltype(g), g)


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


"""
    dualmap(m::AbstractModel, dims::AbstractDualDomain)

Computes both the intensity map and visibility map of the `model`. This can be faster
than computing them separately as some intermediate results can be reused.
This returns a `DualMap` which holds the intensity map and visibility map.
"""
function dualmap(m::AbstractModel, dims::AbstractDualDomain)
    img = intensitymap(m, dims)
    vis = visibilitymap(m, dims)
    map = DualMap(img, vis, dims)
    return map
end

"""
    DualMap(img, vis, dims)

A structure that holds both an image map and a visibility map along with their dual domain.

To access the image map use `imgmap(dm)`.
To access the visibility map use `vismap(dm)`.
To access the dual domain use `domain(dm)`.
"""
struct DualMap{I, V, D <: AbstractDualDomain}
    img::I
    vis::V
    dims::D
end

imgmap(dm::DualMap) = dm.img
vismap(dm::DualMap) = dm.vis
domain(dm::DualMap) = dm.dims


include("lazygrid.jl")
include("executors.jl")
include("headers.jl")
include("rectigrid.jl")
include("structured.jl")


# Define some helpful names for ease typing
