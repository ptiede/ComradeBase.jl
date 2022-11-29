# In this file we will define our base image class. This is entirely based on
# AxisKeys.jl. The specialization occurs in the keys themselves for which we define
# a special type. This type is coded to work similar to a Tuple.


# Define some helpful names for ease typing
const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)}, <:NamedTuple{(:X,:Y)}}

"""
    $(TYPEDEF)

This struct holds the dimensions that the EHT expect. The first type parameter `N`
defines the names of each dimension. These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.
# Fieldnames
$(FIELDS)

# Notes
Warning it is rare you need to access this constructor directly. Instead
use the direct [`IntensityMap`](@ref) function.
"""
struct ImageDimensions{N,T,H}
    """
    Holds the dimensions, usually as a tuple
    """
    dims::T
    """
    A ancillary header that holds additional information about the source,
    i.e. RA, DEC, MJD, other FITS header stuff
    """
    header::H
end

"""
    ImageDimensions{Na}(dims::Tuple, header=nothing) where {Na}

Builds the EHT image dimensions using the names `Na` and dimensions `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.

The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.

# Notes
Instead use the direct [`IntensityMap`](@ref) function.

```julia
dims = ImageDimensions{(:X, :Y, :T, :F)}((-5.0:0.1:5.0, -4.0:0.1:4.0, [1.0, 1.5, 1.75], [230, 345]))
```
"""
@inline function ImageDimensions{Na}(dims::Tuple, header=nothing) where {Na}
    @assert length(Na) == length(dims) "The length of names has to equal the number of dims"
    return ImageDimensions{Na, typeof(dims), typeof(header)}(dims, header)
end

"""
    ImageDimensions(dims::NamedTuple{Na}, header=nothing)

Builds the EHT image dimensions using the names `Na` and dimensions are the values of `dims`.
You can also optionally has a header that stores additional information from e.g.,
a FITS header.

The type parameter `Na` defines the names of each dimension.
These names are usually one of
    - (:X, :Y, :T, :F)
    - (:X, :Y, :F, :T)
    - (:X, :Y) # spatial only
where `:X,:Y` are the RA and DEC spatial dimensions respectively, `:T` is the
the time direction and `:F` is the frequency direction.

# Notes
Instead use the direct [`IntensityMap`](@ref) function.

```julia

dims = ImageDimensions((X=-5.0:0.1:5.0, Y=-4.0:0.1:4.0, T=[1.0, 1.5, 1.75], F=[230, 345]))
```
"""
@inline function ImageDimensions(nt::NamedTuple{N}, header=nothing) where {N}
    dims = values(nt)
    return ImageDimensions{N, typeof(dims), typeof(header)}(dims, header)
end


"""
    dims(d::ImageDimensions)

Returns the dimensions as a tuple of image dimensions
"""
dims(d::ImageDimensions) = d.dims

"""
    named_dims(d::ImageDimensions)

Returns the dimensions as a tuple of image dimensions
"""
named_dims(d::ImageDimensions{Na}) where {Na} = NamedTuple{Na}(d.dims)
Base.keys(::ImageDimensions{Na}) where {Na} = Na
# AxisKeys.getkey(A::IntensityMap; kw...)

# Make ImageDimensions act like a tuple
Base.getindex(d::ImageDimensions, i::Int) = getindex(dims(d), i)
Base.getindex(d::ImageDimensions, inds::Tuple) = getindex(dims(d), inds)
Base.length(d::ImageDimensions) = length(d.dims)
Base.firstindex(d::ImageDimensions) = 1
Base.lastindex(d::ImageDimensions) = length(d)
Base.axes(d::ImageDimensions) = axes(d.dims)
Base.iterate(t::ImageDimensions, i::Int=1) = iterate(t.dims, i)
Base.map(f, d::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(map(f, d.dims), d.header)
Base.front(d::ImageDimensions) = Base.front(d.dims)
Base.eltype(d::ImageDimensions) = eltype(d.dims)


# Now we define our custom image type which is a parametric KeyedArray with a specific key type.
const IntensityMap{T,N,Na} = KeyedArray{T,N,<:AbstractArray{T,N}, <:ImageDimensions{Na}}

AxisKeys.axiskeys(img::IntensityMap{T,1}, d::Int) where {T} = d==1 ? getindex(getfield(img, :keys),1) : OneTo(1)
AxisKeys.axiskeys(img::IntensityMap{T,1}) where {T} = dims(getfield(img, :keys))
# This to make sure that broadcasting and map preverse the keys type
AxisKeys.unify_longest(x::ImageDimensions) = x
AxisKeys.unify_longest(x::ImageDimensions{Na}, y::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(dims(x), dims(y)), x.header)
AxisKeys.unify_longest(x::ImageDimensions{Na}, y::Tuple) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(dims(x), y), x.header)
AxisKeys.unify_longest(x::Tuple, y::ImageDimensions{Na}) where {Na} = ImageDimensions{Na}(AxisKeys.unify_longest(x, dims(y)), y.header)

function AxisKeys.unify_keys(left::ImageDimensions{Na}, right::ImageDimensions) where {Na}
    s = map(AxisKeys.unify_one, dims(left), dims(right))
    return ImageDimensions{Na}(Tuple(s))
end

function AxisKeys.unify_keys(left::ImageDimensions{Na}, right::Tuple) where {Na}
    return ImageDimensions{Na}(AxisKeys.unify_keys(dims(left), right))
end

function AxisKeys.unify_keys(left::Tuple, right::ImageDimensions{Na}) where {Na}
    return ImageDimensions{Na}(AxisKeys.unify_keys(left, dims(right)))
end


AxisKeys.unify_keys(left::ImageDimensions) = left


AxisKeys.named_axiskeys(img::IntensityMap) = named_dims(axiskeys(img))

for (get_or_view, key_get, maybe_copy) in [
    (:getindex, :(AxisKeys.keys_getindex), :copy),
    (:view, :(AxisKeys.keys_view), :identity)
    ]
    @eval begin
        function Base.$get_or_view(A::IntensityMap, raw_inds...)
            inds = to_indices(A, raw_inds)
            @boundscheck checkbounds(parent(A), inds...)
            data = @inbounds $get_or_view(parent(A), inds...)
            inds isa Tuple{Vararg{Integer}} && return data # scalar output

            raw_keys = $key_get(axiskeys(A), inds)
            raw_keys === () && return data # things like A[A .> 0]

            new_keys = ntuple(ndims(data)) do d
                raw_keys === nothing && return axes(data, d)
                raw_keys[d]
            end
            KeyedArray(data, ImageDimensions{dimnames(data)}(new_keys, getfield(A,:keys).header))

        end

        @inline function $key_get(keys::ImageDimensions, inds)
            $key_get(keys.dims, inds)
        end
    end
end



"""
    grid(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is unallocating
since it lazily computes the grid. The grid is an example of a KeyedArray and works similarly.
This is useful for broadcasting a model across an abritrary grid.
"""
RectiGrids.grid(k::IntensityMap) = grid(named_axiskeys(k))

# This is a special constructor to work with ImageDimensions.
function AxisKeys.KeyedArray(data::AbstractArray{T,N}, keys::ImageDimensions{Na}) where {T,N,Na}
    AxisKeys.construction_check(data, keys.dims)
    a = NamedDimsArray(data, Na)
    return KeyedArray{T,N,typeof(a), typeof(keys)}(data, keys)
end

"""
    IntensityMap(data::AbstractArray, dims::NamedTuple)

Constructs an intensitymap using the image dimensions given by `dims`. This returns a
`KeyedArray` with keys given by an `ImageDimensions` object. You can optionally pass a
header object that hold additional information about the image, e.g., a FITS header.

```julia
dims = (X=range(-10.0, 10.0, length=100), Y = range(-10.0, 10.0, length=100),
        T = [0.1, 0.2, 0.5, 0.9, 1.0], F = [230e9, 345e9]
        )
imgk = IntensityMap(rand(100,100,5,1), dims)
```
"""
function IntensityMap(data::AbstractArray{T,N}, dims::NamedTuple{Na,<:NTuple{N,Any}}, header=nothing) where {T,N,Na}
    deht = ImageDimensions(dims, header)
    return KeyedArray(data, deht)
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
                              dims::DataNames, header=nothing
                              ) where {M<:AbstractModel}
    return intensitymap(imanalytic(M), s, dims, header)
end

function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real, y0::Real)
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"

    psizex=fovx/nx
    psizey=fovy/ny

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)

    return (X=xitr, Y=yitr)
end

imagepixels(img::IntensityMap) = (X=img.X, Y=img.Y)

function fieldofview(img::IntensityMap)
    return fieldofview(named_axiskeys(img))
end

function fieldofview(dims::DataNames)
    (;X,Y) = dims
    dx = step(X)
    dy = step(Y)
    (X=abs(last(X) - first(X))+dx, Y=abs(last(Y)-first(Y))+dy)
end

function pixelsizes(img::IntensityMap)
    keys = imagepixels(img)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end

"""
    intensitymap(s, fovx, fovy, nx, ny, x0=0.0, y0=0.0; frequency=230:230, time=0.0:0.0)
"""
function intensitymap(s, fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real=0.0, y0::Real=0.0; frequency=nothing, time=nothing, header=nothing)
    (;X, Y) = imagepixels(fovx, fovy, nx, ny, x0, y0)
    return intensitymap(s, (X=X, Y=Y), header)
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

function intensitymap(::IsAnalytic, s,
                      dims::NamedTuple{N}, header=nothing) where {N}
    dx = step(dims.X)
    dy = step(dims.Y)
    img = intensity_point.(Ref(s), grid(dims)).*dx.*dy
    return IntensityMap(baseimage(img), dims, header)
end


function intensitymap!(::IsAnalytic, img::IntensityMap, s)
    dx, dy = pixelsizes(img)
    g = grid(img)
    img .= intensity_point.(Ref(s), g).*dx.*dy
    return img
end
