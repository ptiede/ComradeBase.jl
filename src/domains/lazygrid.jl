"""
    LazyGrid(dirs::NamedTuple, transform::SMatrix{2, 2})

The lazy array of points of a rectilinear grid with the lookups `dirs`. Element `I` is a
`NamedTuple` with the names of `dirs`: the first two entries are `transform` applied to the
first two lookup values at `I`, the other entries are the lookup values themselves.
"""
struct LazyGrid{T, N, Dirs <: NamedTuple, TR <: SMatrix{2, 2}} <: AbstractArray{T, N}
    dirs::Dirs
    transform::TR
    @inline function LazyGrid(dirs::NamedTuple, transform)
        T = _pointtype(typeof(dirs), eltype(transform))
        return new{T, length(dirs), typeof(dirs), typeof(transform)}(dirs, transform)
    end
end

@inline function _pointtype(::Type{<:NamedTuple{K, A}}, ::Type{R}) where {K, A, R}
    ts = map(eltype, fieldtypes(A))
    S = promote_type(R, ts[1], ts[2])
    return NamedTuple{K, Tuple{S, S, Base.tail(Base.tail(ts))...}}
end

function shapedims(dims::Tuple)
    N = length(dims)
    return ntuple(Val(N)) do n
        Base.@_inline_meta
        reshape(dims[n], ntuple(i -> i == n ? Base.Colon() : 1, Val(N)))
    end
end

function shapedims(dims::NamedTuple{N}) where {N}
    return NamedTuple{N}(shapedims(values(dims)))
end


Base.size(g::LazyGrid) = values(map(length, g.dirs))

function apply_transform(rot::SMatrix{2, 2}, pos::Tuple)
    xy = rot * SVector{2}(pos[1], pos[2])
    return (xy[1], xy[2], Base.tail(Base.tail(pos))...)
end

Base.@propagate_inbounds function Base.getindex(A::LazyGrid{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    pos = map(rgetindex, values(A.dirs), I)
    return T(apply_transform(A.transform, pos))
end

@inline getstyle() = Broadcast.DefaultArrayStyle{0}()
@inline function getstyle(A, Arest...)
    return Broadcast.result_style(Broadcast.BroadcastStyle(A), getstyle(Arest...))
end

function Base.Broadcast.BroadcastStyle(::Type{<:LazyGrid{T, N, A}}) where {T, N, A}
    inner_style = getstyle(fieldtypes(A)...)
    style = Base.Broadcast.result_style(inner_style, Base.Broadcast.DefaultArrayStyle{N}())
    return style
end
