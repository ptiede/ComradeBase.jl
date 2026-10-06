using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format, DimTuple,
    Dimension, XDim, YDim, ZDim, X, Y, Ti

DD.@dim Fr ZDim "frequency"
DD.@dim U XDim "U"
DD.@dim V YDim "V"
DD.@dim Stokes "Stokes"
DD.@dim Fa "feed a"
DD.@dim Fb "feed b"

"""
    Stokes

The polarization dim of a [`StokesMap`](@ref): a trailing dim of length 4 holding Stokes
I, Q, U, V in that order.
"""
Stokes

"""
    Fa

The feed dim of antenna a in a [`CoherencyMap`](@ref), of length 2.
"""
Fa

"""
    Fb

The feed dim of antenna b in a [`CoherencyMap`](@ref), of length 2.
"""
Fb

export IntensityMap, StokesMap, CoherencyMap, Fr, X, Y, Ti, U, V, Stokes, Fa, Fb, eldims,
    coherency, coherencymap, stokesmap, coherencymap!, stokesmap!, spatialdims

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
  4. A polarized map (a [`StokesMap`](@ref)) has the dims of its domain followed by a
     trailing [`Stokes`](@ref) dim of length 4; [`axisdims`](@ref) is the domain and
     [`eldims`](@ref) the trailing dims. Selecting one `Stokes` index gives an unpolarized map
     over the same domain, and indexing that drops every domain dim returns a `DimArray`.
     A coherency map (a [`CoherencyMap`](@ref)) instead has the two trailing feed dims
     [`Fa`](@ref) and [`Fb`](@ref); selecting one index of both gives an unpolarized map, and
     selecting one index of only one of them returns a `DimArray`.

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
struct IntensityMap{T, N, D <: Tuple, G <: AbstractSingleDomain, A <: AbstractArray{T, N}, R <: Tuple, Na, E <: Tuple} <: AbstractDimArray{T, N, D, A}
    data::A
    grid::G
    refdims::R
    name::Na
    eldims::E
    function IntensityMap{T, N}(data, grid, refdims, name, eldims) where {T, N}
        _check_mapsize(data, grid, eldims)
        D = typeof((dims(grid)..., eldims...))
        return new{T, N, D, typeof(grid), typeof(data), typeof(refdims), typeof(name), typeof(eldims)}(
            data, grid, refdims, name, eldims
        )
    end
end

const RectiMap{T, N} = IntensityMap{T, N, <:Tuple, <:AbstractRectiGrid}
const StructuredMap{T, N} = IntensityMap{T, N, <:Tuple, <:StructuredDomain}

"""
    StokesMap{T, N}

An `N`-dimensional `IntensityMap` with element type `T` whose dims are the dims of its domain
followed by a [`Stokes`](@ref) dim of length 4 holding Stokes I, Q, U, V in that order.
Base reductions and element-wise functions act on plain numbers across all dims, Stokes
included; [`flux`](@ref) sums each component and [`stokes`](@ref) selects one.
"""
const StokesMap{T, N} = IntensityMap{T, N, <:Tuple, <:AbstractSingleDomain, <:AbstractArray{T, N}, <:Tuple, <:Any, <:Tuple{Stokes}}

"""
    CoherencyMap{T, N}

An `N`-dimensional `IntensityMap` with element type `T` whose dims are the dims of its domain
followed by the feed dims [`Fa`](@ref) and [`Fb`](@ref), each of length 2. The entry
`[..., a, b]` is the element `e_ab` of the 2×2 coherency matrix at that point: row `a` is the
feed of antenna a and column `b` the feed of antenna b, as in the fields `e11, e21, e12, e22`
of a `CoherencyMatrix`. The dims carry no polarization basis; [`coherencymap`](@ref) and
[`stokesmap`](@ref) take it as an argument. [`coherency`](@ref) selects one element.
"""
const CoherencyMap{T, N} = IntensityMap{T, N, <:Tuple, <:AbstractSingleDomain, <:AbstractArray{T, N}, <:Tuple, <:Any, <:Tuple{Fa, Fb}}

_stokesdim() = Stokes(DD.NoLookup(Base.OneTo(4)))
_feeddims() = (Fa(DD.NoLookup(Base.OneTo(2))), Fb(DD.NoLookup(Base.OneTo(2))))

_polarizationdims(::IsPolarized) = (_stokesdim(),)
_polarizationdims(::NotPolarized) = ()

function _check_mapsize(data, grid, ::Tuple{})
    size(data) == size(grid) || throw(
        DimensionMismatch(
            "IntensityMap data has size $(size(data)), but the $(nameof(typeof(grid))) has size $(size(grid))"
        )
    )
    return nothing
end
function _check_mapsize(data, grid, eldims::Tuple)
    expected = (size(grid)..., map(length, eldims)...)
    size(data) == expected || throw(
        DimensionMismatch(
            "IntensityMap storage has size $(size(data)), but a domain of size $(size(grid)) with trailing dims $(map(DD.name, eldims)) needs size $expected"
        )
    )
    return nothing
end

"""
    eldims(img::IntensityMap)

Returns the dims of `img` that follow the dims of its domain: `(Stokes(...),)` for a
[`StokesMap`](@ref) and `()` for an unpolarized map.
"""
eldims(img::IntensityMap) = getfield(img, :eldims)

# Splits map dims into the domain dims and the trailing element dims.
_splitdims(::Tuple{}) = ((), ())
_splitdims(ds::Tuple) = _splitlast(Base.front(ds), last(ds))
_splitlast(front, d::Stokes) = (front, (d,))
_splitlast(front, d) = ((front..., d), ())
_splitlast(front, d::Fa) = _notamap((front..., d))
_splitlast(::Tuple{}, d::Fb) = _notamap((d,))
_splitlast(front::Tuple, d::Fb) = _splitfeeds(Base.front(front), last(front), d)
_splitfeeds(front, a::Fa, b::Fb) = (front, (a, b))
_splitfeeds(front, a, b::Fb) = _notamap((front..., a, b))
# A lone feed dim has no map meaning; empty domain dims make the caller return a `DimArray`.
_notamap(ds) = ((), ds)

DD.dims(img::IntensityMap) = (dims(getfield(img, :grid))..., getfield(img, :eldims)...)
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
EnzymeRules.inactive(::typeof(eldims), ::IntensityMap) = nothing

"""
    stokes(img::StokesMap, K::Symbol)

Returns the Stokes `K` component (`:I`, `:Q`, `:U` or `:V`) of `img` as an unpolarized
`IntensityMap` over the same domain that is a view of `img`. Every component view has the
same type, so the call infers for any `K`.
"""
@inline function stokes(img::StokesMap, K::Symbol)
    return view(img, Stokes(_stokesindex(K)))
end

@inline function _stokesindex(K::Symbol)
    K === :I && return 1
    K === :Q && return 2
    K === :U && return 3
    K === :V && return 4
    throw(ArgumentError("`$K` is not a Stokes component; the components are I, Q, U, V"))
end

function _broadcast_pointmap!(img::StokesMap, f::F, d) where {F}
    for k in 1:4
        img[Stokes(k)] .= _pointbroadcast(ComponentFn(f, k), d)
    end
    return nothing
end

_stokesI(img::IntensityMap) = _stokesI(img, eldims(img))
_stokesI(img, ::Tuple{}) = img
_stokesI(img, ::Tuple{Stokes}) = stokes(img, :I)
_stokesI(img, eldims::Tuple) = _unsupported_eldims("a Stokes I component", eldims)

function _unsupported_eldims(f, eldims)
    throw(
        ArgumentError(
            "$f is not defined for a map with trailing dims $(map(DD.name, eldims)); for a CoherencyMap convert with `stokesmap` first"
        )
    )
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
and a name for array `name`. An array of `StokesParams` is copied into the storage of a
[`StokesMap`](@ref).
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
    return _wrapstorage(data, grid, refdims, name, ())
end

function IntensityMap(data::AbstractArray{<:StokesParams}, grid::AbstractSingleDomain, refdims::Tuple, name)
    return _wrapstorage(_stokesstorage(data), grid, refdims, name, (_stokesdim(),))
end

"""
    IntensityMap(storage::AbstractArray, d::AbstractSingleDomain, ::Stokes; refdims=(), name=Symbol(""))

Creates a [`StokesMap`](@ref) over `d` whose storage is `storage`, an array of size
`(size(d)..., 4)` holding Stokes I, Q, U, V along its last dim. `storage` is not copied.
"""
function IntensityMap(
        storage::AbstractArray, d::AbstractSingleDomain, ::Stokes;
        refdims = (), name = Symbol("")
    )
    return _wrapstorage(storage, d, refdims, name, (_stokesdim(),))
end

"""
    IntensityMap(storage::AbstractArray, d::AbstractSingleDomain, ::Fa, ::Fb; refdims=(), name=Symbol(""))

Creates a [`CoherencyMap`](@ref) over `d` whose storage is `storage`, an array of size
`(size(d)..., 2, 2)` whose entry `[..., a, b]` is the coherency element `e_ab`. `storage` is
not copied.
"""
function IntensityMap(
        storage::AbstractArray, d::AbstractSingleDomain, ::Fa, ::Fb;
        refdims = (), name = Symbol("")
    )
    return _wrapstorage(storage, d, refdims, name, _feeddims())
end

function _wrapstorage(storage, grid, refdims, name, eldims)
    return IntensityMap{eltype(storage), ndims(storage)}(storage, grid, refdims, name, eldims)
end

_rewrap(img::IntensityMap, storage) = _wrapstorage(storage, axisdims(img), refdims(img), DD.name(img), eldims(img))

function _stokesstorage(x::AbstractArray{<:StokesParams})
    return cat(map(K -> getproperty.(x, K), (:I, :Q, :U, :V))...; dims = Val(ndims(x) + 1))
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

"""
    baseimage(img::IntensityMap)

Returns the storage array of `img`. For a [`StokesMap`](@ref) this is the dense array whose
last dim is `Stokes`.
"""
baseimage(img::IntensityMap) = getfield(img, :data)

@inline function DD.rebuild(
        img::IntensityMap, data, dims::Tuple = dims(img),
        refdims = refdims(img),
        n = name(img),
        metadata = metadata(img),
        executor = executor(img),
    )
    domaindims, eldims = _splitdims(dims)
    isempty(domaindims) && return DD.DimArray(data, dims; refdims, name = n, metadata)
    grid = _rebuild_domain(axisdims(img), domaindims, executor, metadata)
    return _wrapstorage(data, grid, refdims, n, eldims)
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
    I1 = to_indices(img, I)
    N = ndims(axisdims(img))
    domainI = ntuple(k -> I1[k], Val(N))
    return _rebuildsliced(f, img, data, I1, domainI, name)
end

# Only trailing element dims are indexed, so the domain is kept as it is.
function _rebuildsliced(f, img, data, I1, domainI::Tuple{Vararg{Base.Slice}}, name)
    N = length(domainI)
    elI = ntuple(k -> I1[N + k], Val(length(I1) - N))
    neweldims, newrefdims = DD.slicedims(f, eldims(img), refdims(img), elI)
    domaindims, eldims_ = _splitdims((dims(axisdims(img))..., neweldims...))
    if isempty(domaindims)
        newdims = (dims(axisdims(img))..., neweldims...)
        return DD.DimArray(data, newdims; refdims = newrefdims, name, metadata = metadata(img))
    end
    return _wrapstorage(data, axisdims(img), newrefdims, name, eldims_)
end
_rebuildsliced(f, img, data, I1, domainI, name) = _slicemap(f, img, data, I1, domainI, name)

_slicemap(f, img::RectiMap, data, I1, domainI, name) = rebuild(img, data, DD.slicedims(f, img, I1)..., name)

function _slicemap(f, img::StructuredMap, data, I1, domainI, name)
    newdims, newrefdims = DD.slicedims(f, img, I1)
    domaindims, eldims = _splitdims(newdims)
    if !DD.hasdim(domaindims, Pt)
        return DD.DimArray(data, newdims; refdims = newrefdims, name, metadata = metadata(img))
    end
    return _wrapstorage(data, _slice_domain(f, axisdims(img), domainI, domaindims), newrefdims, name, eldims)
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
