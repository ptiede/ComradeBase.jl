using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format, DimTuple,
    Dimension, XDim, YDim, ZDim, X, Y, Ti

DD.@dim Fr ZDim "frequency"
DD.@dim U XDim "U"
DD.@dim V YDim "V"
export IntensityMap, StokesMap, CoherencyMap, Fr, X, Y, Ti, U, V,
    coherency, coherencymap, stokesmap, spatialdims

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
  4. The dims of a map are the dims of its domain, whatever its element type. A polarized map
     (a [`StokesMap`](@ref)) has `StokesParams` elements and a coherency map (a
     [`CoherencyMap`](@ref)) 2×2 matrix elements; [`stokes`](@ref) and [`coherency`](@ref)
     return one component as a map over the same domain.

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
struct IntensityMap{T, N, D <: Tuple, G <: AbstractSingleDomain, A <: AbstractArray{T, N}, R <: Tuple, Na} <: AbstractDimArray{T, N, D, A}
    data::A
    grid::G
    refdims::R
    name::Na
    function IntensityMap{T, N}(data, grid, refdims, name) where {T, N}
        size(data) == size(grid) || throw(
            DimensionMismatch(
                "IntensityMap data has size $(size(data)), but the $(nameof(typeof(grid))) has size $(size(grid))"
            )
        )
        D = typeof(dims(grid))
        return new{T, N, D, typeof(grid), typeof(data), typeof(refdims), typeof(name)}(data, grid, refdims, name)
    end
end

const RectiMap{T, N} = IntensityMap{T, N, <:Tuple, <:AbstractRectiGrid}
const StructuredMap{T, N} = IntensityMap{T, N, <:Tuple, <:StructuredDomain}

"""
    StokesMap{T, N}

An `N`-dimensional `IntensityMap` whose elements are `StokesParams{T}`. The data can be any
array of `StokesParams`; ComradeBase allocates a `ViewStructArray` over dense storage of size
`(size(domain)..., 4)` holding Stokes I, Q, U, V along its last dim. [`flux`](@ref) sums the
elements and [`stokes`](@ref) selects one component.
"""
const StokesMap{T, N} = IntensityMap{StokesParams{T}, N}

"""
    CoherencyMap{T, N}

An `N`-dimensional `IntensityMap` whose elements are 2×2 coherency matrices
`SMatrix{2, 2, T, 4}`: row `a` is the feed of antenna a and column `b` the feed of antenna b,
as in the fields `e11, e21, e12, e22` of a `CoherencyMatrix`. The elements carry no
polarization basis; [`coherencymap`](@ref) and [`stokesmap`](@ref) take it as an argument.
ComradeBase allocates a `ViewStructArray` over dense storage of size `(size(domain)..., 2, 2)`.
[`coherency`](@ref) selects one element.
"""
const CoherencyMap{T, N} = IntensityMap{SMatrix{2, 2, T, 4}, N}

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

# Resolves the ambiguity between DimensionalData's (DimensionalStyle, AbstractArrayStyle) and
# StructArrays' (AbstractArrayStyle, StructArrayStyle) rules, keeping the dims outermost.
function Base.BroadcastStyle(
        ::DimensionalData.DimensionalStyle{A}, b::StructArrays.StructArrayStyle
    ) where {A}
    return DimensionalData.DimensionalStyle(A(), b)
end


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

"""
    stokes(img::StokesMap, K::Symbol)

Returns the Stokes `K` component (`:I`, `:Q`, `:U` or `:V`) of `img` as an unpolarized
`IntensityMap` over the same domain. For `ViewStructArray` and `StructArray` data this is a
view of the data; other arrays are copied.
"""
@inline function stokes(img::StokesMap, K::Symbol)
    return IntensityMap(stokes(baseimage(img), K), axisdims(img), refdims(img), DD.name(img))
end

function Base.propertynames(img::IntensityMap)
    return propertynames(axisdims(img))
end

@inline Base.@constprop :aggressive function Base.getproperty(img::IntensityMap, p::Symbol)
    return getproperty(axisdims(img), p)
end

const SpatialDims = Tuple{<:DD.Dimensions.X, <:DD.Dimensions.Y}
const SpatialIntensityMap{T, G <: AbstractRectiGrid, A} = IntensityMap{T, 2, <:SpatialDims, G, A}

"""
    spatialdims(g::AbstractRectiGrid)
    spatialdims(img::IntensityMap)

Return the sub-grid spanned by the *first two* dimensions of `g`, dropping any additional
dimensions such as frequency (`Fr`) or time (`Ti`).

Grids here place the two spatial dimensions first (see `SpatialDims`), so those two
are `X` and `Y` for any conventionally constructed grid. The names are not checked:
whatever the first two dimensions are is what you get back.
"""
function spatialdims(g::AbstractRectiGrid)
    ds = dims(g)
    return rebuild(g; dims = ds[1:2])
end
spatialdims(img::IntensityMap) = spatialdims(axisdims(img))

"""
    IntensityMap(data::AbstractArray, g::AbstractSingleDomain; refdims=(), name=Symbol(""))

Creates an `IntensityMap` with the values `data` on the domain `g`, e.g. pixel fluxes on a
[`RectiGrid`](@ref) or values at the points of a [`StructuredDomain`](@ref). `size(data)` must
equal `size(g)`. Optionally, you can specify a set of reference dimensions `refdims` as a tuple
and a name for array `name`. `data` is not copied; an array of `StokesParams` gives a
[`StokesMap`](@ref), e.g. `IntensityMap(ViewStructArray{StokesParams}(P), g)` for dense storage
`P` of size `(size(g)..., 4)`.
"""
function IntensityMap(
        data::AbstractArray, g::AbstractSingleDomain; refdims = (),
        name = Symbol("")
    )
    return IntensityMap(data, g, refdims, name)
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

function IntensityMap(data::IntensityMap, g::AbstractSingleDomain)
    g == axisdims(data) || throw(
        ArgumentError("the domain of the IntensityMap is not the $(nameof(typeof(g))) given")
    )
    return data
end

function IntensityMap(data::AbstractArray, grid::AbstractSingleDomain, refdims::Tuple, name)
    return IntensityMap{eltype(data), ndims(data)}(data, grid, refdims, name)
end

_rewrap(img::IntensityMap, data) = IntensityMap(data, axisdims(img), refdims(img), DD.name(img))

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

"""
    baseimage(img::IntensityMap)

Returns the data array of `img`. For a map over a `ViewStructArray` the dense storage is
`parent(baseimage(img))`.
"""
baseimage(img::IntensityMap) = getfield(img, :data)

@inline function DD.rebuild(
        img::IntensityMap, data, dims::Tuple = dims(img),
        refdims = refdims(img),
        n = name(img),
        metadata = metadata(img),
        executor = executor(img),
    )
    isempty(dims) && return DD.DimArray(data, dims; refdims, name = n, metadata)
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
        f::Function, img::IntensityMap, data::AbstractArray, I::Tuple, name = DD.name(img)
    )
    return _slicemap(f, img, data, to_indices(img, I), name)
end

_slicemap(f, img::RectiMap, data, I, name) = rebuild(img, data, DD.slicedims(f, img, I)..., name)

function _slicemap(f, img::StructuredMap, data, I, name)
    newdims, newrefdims = DD.slicedims(f, img, I)
    if !DD.hasdim(newdims, Pt)
        return DD.DimArray(data, newdims; refdims = newrefdims, name, metadata = metadata(img))
    end
    return IntensityMap(data, _slice_domain(f, axisdims(img), I, newdims), newrefdims, name)
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

function intensitymap_analytic_executor!(img::IntensityMap, s::AbstractModel, executor)
    g = axisdims(img)
    _pointmap!(img, _intensityfn(s, g), g, executor)
    return nothing
end

"""
    PixelFlux(model, area)

Called with a point `p`, returns `intensity_point(model, p) * area`: the flux in a pixel of
that area.
"""
struct PixelFlux{M, T}
    model::M
    area::T
end
@inline (f::PixelFlux)(p) = intensity_point(f.model, p) * f.area

_intensityfn(s, ::StructuredDomain) = Base.Fix1(intensity_point, s)
_intensityfn(s, g::AbstractRectiGrid) = PixelFlux(s, prod(pixelsizes(g)))
