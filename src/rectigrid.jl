export RectiGrid

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
    r = (c, -s, s, c)
    m = SMatrix{2,2,typeof(s),4}(r)
    return m
end

function domainpoints(d::RectiGrid{D,Hd}) where {D,Hd}
    g = map(basedim, dims(d))
    rot = rotmat(d)
    N = keys(d)
    return RotGrid(StructArray(NamedTuple{N}(_build_slices(g, size(d)))), rot)
end

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

!!! warn
    The `posang` argument and the overall rotation of the grid is currently experimental and
    and may change abruptly in the future even on minor releases.

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
