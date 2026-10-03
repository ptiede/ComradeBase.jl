using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format, DimTuple,
    Dimension, XDim, YDim, ZDim, X, Y, Ti

DD.@dim Fr ZDim "frequency"
DD.@dim U XDim "U"
DD.@dim V YDim "V"

export IntensityMap, Fr, X, Y, Ti, U, V

"""
    $(TYPEDEF)

This type is the basic array type for all images and models that obey the `ComradeBase`
interface. The type is a subtype of `DimensionalData.AbstractDimArray` however, we make
a few changes to support the Comrade API.

  1. The dimensions are given by a domain: an `AbstractRectiGrid` (usually [`RectiGrid`](@ref))
     for images on rectilinear grids, or a [`StructuredDomain`](@ref) for values at points
     such as visibilities. Operations that need pixels (`pixelsizes`, `fieldofview`, `flux`,
     `centroid`, `second_moment`, `phasecenter`) are defined only for rectilinear grids.
  2. There are two ways to access the dimensions of the array. `dims(img)` will
     return the usual `DimArray` dimensions, i.e. a `Tuple{DimensionalData.Dim, ...}`.
     The other way to access the array dimensions is using the `getproperty`, e.g.,
     `img.X` will return the RA/X grid locations but stripped of the usual `DimensionalData.Dimension`
     material. This `getproperty` behavior is *NOT CONSIDERED** part of the stable API and
     may be changed in the future.
  3. Metadata is stored in the domain through the `header` property and can be
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
For a map over a `StructuredDomain`, indexing slices the coordinates along with the values;
indexing that drops the `Pt` dim returns a `DimArray`, and any other operation that changes
the dim names or sizes (e.g. a reduction over `dims`) throws an `ArgumentError`, so apply it
to `DimArray(img)` instead.
"""
struct IntensityMap{T, N, D <: Tuple, G <: AbstractSingleDomain{D}, A <: AbstractArray{T, N}, R <: Tuple, Na} <: AbstractDimArray{T, N, D, A}
    data::A
    grid::G
    refdims::R
    name::Na
    function IntensityMap(
            data::A, grid::G, refdims::R,
            name::Na
        ) where {
            A <: AbstractArray{T, N}, G <: AbstractSingleDomain{D},
            R <: Tuple, Na,
        } where {T, N, D <: Tuple}
        _check_mapsize(data, grid)
        return new{T, N, D, G, A, R, Na}(data, grid, refdims, name)
    end
end

const RectiMap{T, N} = IntensityMap{T, N, <:Tuple, <:AbstractRectiGrid}

_check_mapsize(data, grid) = nothing
function _check_mapsize(data, grid::StructuredDomain)
    size(data) == size(grid) || throw(
        DimensionMismatch(
            "IntensityMap data has size $(size(data)), but the StructuredDomain has size $(size(grid))"
        )
    )
    return nothing
end

DD.dims(img::IntensityMap) = dims(getfield(img, :grid))
DD.refdims(img::IntensityMap) = getfield(img, :refdims)
DD.data(img::IntensityMap) = getfield(img, :data)
DD.name(img::IntensityMap) = getfield(img, :name)
DD.metadata(img::IntensityMap) = header(axisdims(img))
executor(img::IntensityMap) = executor(axisdims(img))

# TODO add this to DimensionalData directly
EnzymeRules.inactive(::typeof(DD._broadcasted_dims), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DD.Dimensions.comparedims), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DD.Dimensions._comparedims), args...; kwargs...) = nothing


# DD 0.30 regression: DimensionalStyle{StructArrayStyle,N} ⊕ DefaultArrayStyle{0}
# resolves to Unknown via StructArrays' deliberate fallback, dropping the
# DimensionalStyle and producing a DimArray instead of an IntensityMap.
# A scalar broadcast shouldn't change the outer style, so keep ours.
Base.BroadcastStyle(
    s::DimensionalData.DimensionalStyle{<:StructArrays.StructArrayStyle, N},
    ::Base.Broadcast.DefaultArrayStyle{M},
) where {N, M} = s


# We need this to make sure IntensityMap works correctly on the GPU
# function Base.copyto!(dest::IntensityMap, bc::Broadcast.Broadcasted)
#     copyto!(baseimage(dest), bc)
#     return dest
# end

# For the `IntensityMap` nothing is AD-able except the data so
# let's tell Enzyme this
EnzymeRules.inactive(::typeof(DD.dims), ::IntensityMap) = nothing
EnzymeRules.inactive(::typeof(DD.refdims), ::IntensityMap) = nothing
EnzymeRules.inactive(::typeof(DD.name), ::IntensityMap) = nothing
EnzymeRules.inactive(::typeof(DD.metadata), ::IntensityMap) = nothing
EnzymeRules.inactive(::typeof(executor), ::IntensityMap) = nothing

@inline function stokes(pimg::IntensityMap{<:StokesParams}, v::Symbol)
    return rebuild(pimg; data = stokes(baseimage(pimg), v))
end

function Base.propertynames(img::IntensityMap)
    return propertynames(axisdims(img))
end

@inline Base.@constprop :aggressive function Base.getproperty(img::IntensityMap, p::Symbol)
    return getproperty(axisdims(img), p)
end

const SpatialDims = Tuple{<:DD.Dimensions.X, <:DD.Dimensions.Y}
const SpatialIntensityMap{T, A, G} = IntensityMap{T, 2, <:SpatialDims, A, G} where {T, A <: AbstractRectiGrid, G}

"""
    IntensityMap(data::AbstractArray, g::AbstractRectiGrid; refdims=(), name=Symbol(""))

Creates a IntensityMap with the pixel fluxes `data` on the grid `g`. Optionally, you can specify
a set of reference dimensions `refdims` as a tuple and a name for array `name`.
"""
function IntensityMap(
        data::AbstractArray, g::AbstractRectiGrid; refdims = (),
        name = Symbol("")
    )
    return IntensityMap(data, g, (), Symbol(""))
end

"""
    IntensityMap(data::AbstractArray, fovx::Number, fovy::Number, x0::Number=0, y0::Number=0; header=NoHeader())

Creates a IntensityMap with the pixel fluxes `data` and a spatial grid with field of view
(`fovx`, `fovy`) and center pixel offset (`x0`, `y0`) and header `header`.
"""
function IntensityMap(
        data::AbstractArray{T}, fovx::Number, fovy::Number, x0::Number = 0,
        y0::Number = 0; header = NoHeader()
    ) where {T}
    grid = imagepixels(fovx, fovy, size(data)..., T(x0), T(y0); header)
    return IntensityMap(data, grid)
end

function IntensityMap(data::IntensityMap, g::AbstractRectiGrid)
    @assert g == axisdims(data) "Dimensions do not agree"
    return data
end

"""
    IntensityMap(data::AbstractArray, d::StructuredDomain; refdims=(), name=Symbol(""))

Creates an `IntensityMap` with the values `data` at the points of `d`. `size(data)` must
equal `size(d)`.
"""
function IntensityMap(
        data::AbstractArray, d::StructuredDomain; refdims = (),
        name = Symbol("")
    )
    return IntensityMap(data, d, refdims, name)
end

function IntensityMap(data::IntensityMap, d::StructuredDomain)
    d == axisdims(data) || throw(
        ArgumentError("the domain of the IntensityMap is not the StructuredDomain given")
    )
    return data
end

"""
    axisdims(img::IntensityMap)
    axisdims(img::IntensityMap, p::Symbol)

Returns the keys of the `IntensityMap` as the actual internal `AbstractRectiGrid` object.
Optionall the user can ask for a specific dimension with `p`
"""
@inline axisdims(img::IntensityMap) = getfield(img, :grid)
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
        executor = executor(img),
    )
    grid = _rebuild_domain(axisdims(img), dims, executor, metadata)
    return IntensityMap(data, grid, refdims, n)
end

function _rebuild_domain(g::AbstractRectiGrid, dims, executor, metadata)
    return rebuild(g, dims, executor, metadata, posang(g))
end

function _rebuild_domain(g::StructuredDomain, dims, executor, metadata)
    names = map(DD.name, dims)
    sizes = map(length, dims)
    (names == keys(g) && sizes == size(g)) || throw(
        ArgumentError(
            "an IntensityMap over a StructuredDomain with dims $(keys(g)) and size $(size(g)) cannot take dims $names with size $sizes, since its coordinates do not follow; apply the operation to `DimArray(img)` instead"
        )
    )
    return rebuild(g; dims, executor, header = metadata)
end

Base.@propagate_inbounds function DD.rebuildsliced(
        f::Function, img::IntensityMap{<:Any, <:Any, <:Tuple, <:StructuredDomain},
        data::AbstractArray, I::Tuple, name = DD.name(img)
    )
    I1 = to_indices(img, I)
    newdims, newrefdims = DD.slicedims(f, img, I1)
    d = axisdims(img)
    if !(:Pt in map(DD.name, newdims))
        return DD.DimArray(data, newdims; refdims = newrefdims, name, metadata = metadata(img))
    end
    return IntensityMap(data, _slice_domain(f, d, I1, newdims), newrefdims, name)
end

@inline function DD.rebuild(
        img::IntensityMap;
        data = DD.data(img),
        dims::Tuple = dims(img),
        refdims = refdims(img),
        name = name(img),
        metadata = metadata(img),
        executor = executor(img),
    )
    return rebuild(img, data, dims, refdims, name, metadata, executor)
end

function intensitymap_analytic_executor!(img::RectiMap, s::AbstractModel, ::Serial)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    for I in eachindex(g, bimg)
        bimg[I] = intensity_point(s, g[I]) * dx * dy
    end
    # bimg .= intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function intensitymap_analytic_executor!(
        img::RectiMap, s::AbstractModel,
        ::ThreadsEx{S}
    ) where {S}
    g = domainpoints(img)
    e = executor(img)
    dx, dy = pixelsizes(img)
    @threaded e for I in CartesianIndices(g)
        img[I] = intensity_point(s, g[I]) * dx * dy
    end
    return nothing
end

function _threads_intensitymap! end
