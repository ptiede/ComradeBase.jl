using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format, DimTuple,
                       Dimension, XDim, YDim, ZDim, X, Y, Ti


DD.@dim Fr ZDim "frequency"
DD.@dim U  XDim "U"
DD.@dim V  YDim "V"

export IntensityMap, Fr, X, Y, Ti, U, V

"""
    $(TYPEDEF)

This type is the basic array type for all images and models that obey the `ComradeBase`
interface. The type is a subtype of `DimensionalData.AbstractDimArray` however, we make
a few changes to support the Comrade API.

  1. The dimensions should be specified by an `AbstractRectiGrid` interface. Usually users just
     need the [`RectiGrid`](@ref) grid, for rectilinear grids.
  2. There are two ways to access the dimensions of the array. `dims(img)` will
     return the usual `DimArray` dimensions, i.e. a `Tuple{DimensionalData.Dim, ...}`.
     The other way to access the array dimensions is using the `getproperty`, e.g.,
     `img.X` will return the RA/X grid locations but stripped of the usual `DimensionalData.Dimension`
     material. This `getproperty` behavior is *NOT CONSIDERED** part of the stable API and
     may be changed in the future.
  3. Metadata is stored in the `AbstractRectiGrid` type through the `header` property and can be
     accessed through `metadata` or `header`

The most common way to create a `IntensityMap` is to use the function definitions
```julia-repl
julia> g = imagepixels(10.0, 10.0, 128, 128; header=NoHeader())
julia> X = g.X; Y = g.Y
julia> data = rand(128, 128)
julia> img1 = IntensityMap(data, g)
julia> img2 = IntensityMap(data, (;X, Y); header=header(g))
julia> img1 == img2
true
julia> img3 = IntensityMap(data, 10.0, 10.0; header=NoHeader())
```

Broadcasting, map, and reductions should all just obey the `DimensionalData` interface.
"""
struct IntensityMap{T,N,D,G<:AbstractRectiGrid{D},A<:AbstractArray{T,N},R<:Tuple,Na} <: AbstractDimArray{T,N,D,A}
    data::A
    grid::G
    refdims::R
    name::Na
    function IntensityMap(
        data::A, grid::G, refdims::R, name::Na
        ) where {A<:AbstractArray{T,N}, G<:AbstractRectiGrid{D}, R<:Tuple, Na} where {T,N,D}

        new{T,N,D,G,A,R,Na}(data, grid, refdims, name)
    end
end

DD.dims(img::IntensityMap)    = dims(getfield(img, :grid))
DD.refdims(img::IntensityMap) = getfield(img, :refdims)
DD.data(img::IntensityMap)    = getfield(img, :data)
DD.name(img::IntensityMap)    = getfield(img, :name)
DD.metadata(img::IntensityMap)= header(axisdims(img))

executor(img::IntensityMap)   = executor(axisdims(img))

@inline function stokes(pimg::IntensityMap{<:StokesParams}, v::Symbol)
    IntensityMap(stokes(baseimage(pimg), v), axisdims(pimg), refdims(pimg), name(pimg))
end


function Base.propertynames(img::IntensityMap)
    return keys(axisdims(img))
end

function Base.getproperty(img::IntensityMap, p::Symbol)
    if p ∈ propertynames(img)
        return basedim(getproperty(axisdims(img), p))
    else
        throw(ArgumentError("$p not a valid dimension of `IntensityMap`"))
    end
end

const SpatialDims = Tuple{<:DD.Dimensions.X, <:DD.Dimensions.Y}
const SpatialIntensityMap{T, A, G} = IntensityMap{T,2,<:SpatialDims, A, G} where {T,A,G}

"""
    IntensityMap(data::AbstractArray, g::AbstractRectiGrid; refdims=(), name=Symbol(""))

Creates a IntensityMap with the pixel fluxes `data` on the grid `g`. Optionally, you can specify
a set of reference dimensions `refdims` as a tuple and a name for array `name`.
"""
function IntensityMap(data::AbstractArray, g::AbstractRectiGrid; refdims=(), name=Symbol(""))
    return IntensityMap(data, g, (), Symbol(""))
end

"""
    IntensityMap(data::AbstractArray, fovx::Real, fovy::Real, x0::Real=0, y0::Real=0; header=NoHeader())

Creates a IntensityMap with the pixel fluxes `data` and a spatial grid with field of view
(`fovx`, `fovy`) and center pixel offset (`x0`, `y0`) and header `header`.
"""
function IntensityMap(data::AbstractArray{T}, fovx::Real, fovy::Real, x0::Real=0, y0::Real=0; header=NoHeader()) where {T}
    grid = imagepixels(fovx, fovy, size(data)..., T(x0), T(y0); header)
    return IntensityMap(data, grid)
end


function IntensityMap(data::IntensityMap, g::AbstractRectiGrid)
    @assert g == axisdims(data) "Dimensions do not agree"
    return data
end


"""
    axisdims(img::IntensityMap)
    axisdims(img::IntensityMap, p::Symbol)

Returns the keys of the `IntensityMap` as the actual internal `AbstractRectiGrid` object.
Optionall the user can ask for a specific dimension with `p`
"""
axisdims(img::IntensityMap) = getfield(img, :grid)
axisdims(img::IntensityMap, p::Symbol) = getproperty(axisdims(img), p)
EnzymeRules.inactive(::typeof(axisdims), args...) = nothing
named_dims(img::IntensityMap) = named_dims(axisdims(img))


"""
    header(img::IntensityMap)

Retrieves the header of an IntensityMap
"""
header(img::IntensityMap) = header(axisdims(img))

DD._noname(::IntensityMap) = Symbol("")


Base.parent(img::IntensityMap) = DD.data(img)

baseimage(x::IntensityMap) = baseimage(parent(x))

@inline function DD.rebuild(
    img::IntensityMap, data, dims::Tuple = dims(img),
    refdims = refdims(img),
    n = name(img),
    metadata = metadata(img),
    )
    # @info (typeof(img))
    # TODO find why Name is changing type
    # n2 = n == Symbol("") ? NoName : n
    # @info which(name, (typeof(img),))
    grid = rebuild(typeof(axisdims(img)), dims, executor(img), metadata)
    # return name(img)
    return IntensityMap(data, grid, refdims, n)
end

@inline function DD.rebuild(
    img::IntensityMap; data=DD.data(img), dims::Tuple = dims(img),
    refdims = refdims(img),
    name = name(img),
    metadata = metadata(img),
    )
    rebuild(img, data, dims, refdims, name, metadata)
end

function intensitymap_analytic(s::AbstractModel, dims::AbstractRectiGrid)
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), domainpoints(dims)).*dx.*dy
    return IntensityMap(img, dims)
end

function intensitymap_analytic!(img::IntensityMap, s::AbstractModel)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return nothing
end

function intensitymap_analytic(s::AbstractModel, dims::AbstractRectiGrid{D, <:ThreadsEx}) where {D}
    img = allocate_imgmap(s, dims)
    intensitymap_analytic!(img, s)
    return img
end

function intensitymap_analytic!(
    img::IntensityMap{T,N,D,<:ComradeBase.AbstractRectiGrid{D, <:ThreadsEx{S}}},
    s::AbstractModel) where {T,N,D,S}
    g = domainpoints(img)
    _threads_intensitymap!(img, s, g, Val(S))
    return nothing
end


for s in schedulers
    @eval begin
        function _threads_intensitymap!(img::IntensityMap, s::AbstractModel, g, ::Val{$s})
            dx, dy = pixelsizes(img)
            Threads.@threads $s for I in CartesianIndices(g)
                img[I] = intensity_point(s, g[I])*dx*dy
            end
        end
    end
end
