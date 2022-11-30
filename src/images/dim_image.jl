using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format, DimTuple,
                       Dimension


DD.@dim Fr "frequency"


struct IntensityMap{T,N,D<:Tuple,R<:Tuple,A<:AbstractArray{T,N},Na,Me} <: AbstractDimArray{T,N,D,A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me
end

function IntensityMap(
        data::AbstractArray{T,N}, dims::Union{<:NTuple{N, Dimension}, NamedTuple{Na, NTuple{N}}};
        refdims=(), name=NoName(), metadata=NoMetadata()) where {T,N,Na}

    return IntensityMap(data, dims, refdims, name, metadata)
end

function IntensityMap(; data, dims, refdims=(), name=NoName(), metadata=NoMetadata())
    return IntensityMap(data, dims; refdims, name, metadata)
end


function IntensityMap(A::AbstractDimArray;
    data=data(A), dims=dims(A), refdims=refdims(A), name=name(A), metadata=metadata(A)
)
    return IntensityMap(data, dims; refdims, name, metadata)
end

@inline function DD.rebuild(
        ::IntensityMap, data::AbstractArray,
        dims::Tuple, refdims::Tuple, name, metadata)

    return IntensityMap(data, dims, refdims, name, metadata)
end

const StokesIntensityMap{T,D,N,A} = IntensityMap{StokesParams{T},N,D,<:Tuple,<:Tuple,A}

function check_grid(I,Q,U,V)
    named_dims(I) == named_dims(Q) == named_dims(U) == named_dims(V)
end

function StokesIntensityMap(I::IntensityMap, Q::IntensityMap, U::IntensityMap, V::IntensityMap)
    @assert check_grid(I,Q,U,V) "I,Q,U,V have different griddings/dimensions, this is not supported in StokesIntensityMap"
    pI = parent(I)
    pQ = parent(Q)
    pU = parent(U)
    pV = parent(V)
    return rebuild(I, StructArray{StokesParams{eltype(pI)}}((I=pI, Q=pQ, U=pU, V=pV)))
end

function named_dims(img::IntensityMap)
    d = dims(img)
    return NamedTuple{name(d)}(d)
end

function stokes(pimg::StokesIntensityMap, v::Symbol)
    imgb = parent(pimg)
    imgs = getproperty(imgb, v)
    return rebuild(pimg, imgs)
end

function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real, y0::Real)
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"

    psizex=fovx/nx
    psizey=fovy/ny

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)

    return (X(xitr), Y(yitr))
end

imagepixels(img::IntensityMap) = (X=dims(img, X), Y=dims(img, Y))

function fieldofview(img::IntensityMap)
    x,y = dims(img, X, Y)
    dx = step(x)
    dy = step(y)
    return (X=abs(last(x) - first(x))+dx, Y=abs(last(y)-first(y))+dy)
end


function IntensityMap(data, fovx::Real, fovy::Real, x0::Real = 0.0, y0::Real = 0.0; kw...)
    dims = imagepixels(fovx, fovy, size(data)..., x0, y0)
    return IntensityMap(dta, dims; kw...)
end

"""
    intensitymap(model::AbstractModel, dims, header=nothing)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `KeyedArray` with [`ImageDimensions`](@ref) as keys. The dimensions are a `NamedTuple`
and must have one of the following names:
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
"""
@inline function intensitymap(s::M,
                              dims::Union{DimTuple, NamedTuple}, kw...
                              ) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims, kw...)
end


"""
    intensitymap(s, fovx, fovy, nx, ny, x0=0.0, y0=0.0; frequency=230:230, time=0.0:0.0)
"""
function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0; kwargs...)
    dims = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, dims, kwargs...)
end


"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img::IntensityMap, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end

using DimensionalData
function namedtuple2dim(d::NamedTuple)
    n = DimensionalData.key2dim.(keys(d))
    v = values(d)
    return rebuild.(n, v)
end

function intensitymap(::IsAnalytic, s,
                      dims::NamedTuple; kwargs...)
    d = namedtuple2dim(dims)
    return intensitymap(IsAnalytic(), s, d; kwargs...)
end

struct NamedDimPoints{Na,T,N,D<:NTuple{N,Dimension}} <: DimensionalData.AbstractDimIndices{T,N}
    dims::D
end

export NamedDimPoints
function NamedDimPoints(dims::DimTuple)
    Na = name(dims)
    T = Tuple{map(eltype, dims)...}
    N = length(dims)
    NamedDimPoints{Na,T,N,typeof(dims)}(dims)
end

Base.@propagate_inbounds function Base.getindex(dp::NamedDimPoints{Na,T,N}, I::Vararg{Int,N}) where {Na,T,N}
    dp = getindex.(dp.dims, I)
    return NamedTuple{Na}(dp)
end

Base.size(d::NamedDimPoints) = map(length, d.dims)
Base.length(d::NamedDimPoints) = prod(size(d))


function intensitymap(::IsAnalytic, s, d::DimTuple)
    return IntensityMap(_intensitymap_analytic(s, d), d)
end

@noinline function _intensitymap_analytic(s, d)
    dx = step(dims(d,X))
    dy = step(dims(d,Y))
    grid = NamedDimPoints(d)
    img = intensity_point.(Ref(s), grid).*dx.*dy
    return img
end


function intensitymap!(::IsAnalytic, img::IntensityMap, s)
    dx, dy = pixelsizes(img)
    g = grid(img)
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return img
end
