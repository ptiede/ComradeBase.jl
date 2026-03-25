export RectiGrid, refinespatial

abstract type AbstractRectiGrid{D, E} <: AbstractSingleDomain{D, E} end
create_map(array, g::AbstractRectiGrid) = IntensityMap(array, g)
function allocate_map(M::Type{<:AbstractArray{T}}, g::AbstractRectiGrid) where {T}
    arr = similar(M, size(g))
    return IntensityMap(arr, g)
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


struct RectiGrid{D, E, Hd <: AMeta, P} <: AbstractRectiGrid{D, E}
    dims::D
    executor::E
    header::Hd
    posang::P
    @inline function RectiGrid(
            dims::Tuple; executor = Serial(),
            header::AMeta = NoHeader(), posang = zero(eltype(first(dims)))
        )
        df = DD.format(dims)
        return new{typeof(df), typeof(executor), typeof(header), typeof(posang)}(
            df, executor,
            header,
            posang
        )
    end
end

EnzymeRules.inactive_type(::Type{<:RectiGrid}) = true

function rotmat(d::RectiGrid)
    s, c = sincos(posang(d))
    r = (c, -s, s, c)
    m = SMatrix{2, 2, typeof(s), 4}(r)
    return m
end

function domainpoints(d::RectiGrid{D, Hd}) where {D, Hd}
    g = map(basedim, dims(d))
    rot = rotmat(d)
    N = keys(d)
    return LazyGrid(NamedTuple{N}(g), rot)
end

@inline Base.keys(g::RectiGrid) = map(name, dims(g))

@inline RectiGrid(g::RectiGrid) = g

function Base.show(io::IO, mime::MIME"text/plain", x::RectiGrid{D, E}) where {D, E}
    println(io, "RectiGrid(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    return print(io, "\n)")
end

Base.propertynames(d::RectiGrid) = keys(d)
# This needs to be inlined to avoid performance issues
@inline function Base.getproperty(g::RectiGrid, p::Symbol)
    hasproperty(g, p) || throw(ArgumentError("RectiGrid does not have property $p"))
    return basedim(getproperty(named_dims(g), p))
end

# This is needed to prevent doubling up on the dimension
@inline function RectiGrid(
        dims::NamedTuple{Na, T}; executor = Serial(),
        header::AMeta = NoHeader(),
        posang = zero(eltype(first(values(dims))))
    ) where {
        Na, N,
        T <: NTuple{
            N,
            DD.Dimension,
        },
    }
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

!!! warn
    The `posang` argument and the overall rotation of the grid is currently experimental and
    and may change abruptly in the future even on minor releases.

## Examples

```julia
dims = RectiGrid((X(-5.0:0.1:5.0), Y(-4.0:0.1:4.0), Ti([1.0, 1.5, 1.75]), Fr([230, 345])); executor=ThreadsEx())
dims = RectiGrid((X = -5.0:0.1:5.0, Y = -4.0:0.1:4.0, Ti = [1.0, 1.5, 1.75], Fr = [230, 345]); executor=ThreadsEx()))
```
"""
@inline function RectiGrid(
        nt::NamedTuple; executor = Serial(),
        header::AMeta = ComradeBase.NoHeader(),
        posang = zero(eltype(first(values(nt))))
    )
    dims = _make_dims(keys(nt), values(nt))
    return RectiGrid(dims; executor, header, posang)
end

function DD.rebuild(
        grid::RectiGrid, dims, executor = executor(grid),
        header = metadata(grid), posang = posang(grid)
    )
    return RectiGrid(dims; executor, header, posang)
end

function DD.rebuild(
        grid::RectiGrid; dims = dims(grid), executor = executor(grid),
        header = metadata(grid), posang = posang(grid)
    )
    return rebuild(grid, dims, executor, header, posang)
end

function refinespatial(g::RectiGrid, refac::NTuple{2})
    ns = size(g)[1:2]
    d = map(basedim, named_dims(g))
    d2 = @set d.X.len = ceil(Int, ns[1] * refac[1])
    d3 = @set d2.Y.len = ceil(Int, ns[2] * refac[2])
    dn = dims(g)
    dnn = @set dn[1] = X(d3.X)
    dnn2 = @set dnn[2] = Y(d3.Y)
    dnew = (dnn2[1:2]..., dn[3:end]...)
    return rebuild(g; dims = dnew)
end

refinespatial(g::RectiGrid, refac::Number) = refinespatial(g, (refac, refac))
