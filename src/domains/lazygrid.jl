struct LazyGrid{T, N, Dirs, TR <: Union{Nothing, AbstractMatrix}} <: AbstractArray{T, N}
    dirs::Dirs
    dims::Dims{N}
    transform::TR
    @inline function LazyGrid(
            dirs::Union{NamedTuple, Tuple},
            transform = identity
        )
        T = geteltype(typeof(dirs))
        N = length(dirs)
        dims = values(map(length, dirs))
        return new{T, N, typeof(dirs), typeof(transform)}(dirs, dims, transform)
    end
end

@inline function geteltype(::Type{<:NamedTuple{N, A}}) where {N, A}
    return NamedTuple{N, geteltype(A)}
end

@inline function geteltype(T::Type{<:Tuple})
    et = map(eltype, fieldtypes(T))
    return Tuple{et...}
end

@inline function DD.dims(g::LazyGrid{T, N}) where {T, N}
    dms = g.dirs
    return ntuple(Val(N)) do n
        Base.@_inline_meta
        reshape(dms[n], ntuple(i -> i == n ? Base.Colon() : 1, Val(N)))
    end
end

function shapedims(dims::NTuple{N}) where {N}
    return ntuple(Val(N)) do n
        Base.@_inline_meta
        reshape(dims[n], ntuple(i -> i == n ? Base.Colon() : 1, Val(N)))
    end
end

function shapedims(dims::NamedTuple{N}) where {N}
    return NamedTuple{N}(shapedims(dims))
end


Base.size(g::LazyGrid) = g.dims

function apply_transform(::Nothing, pos)
    return pos
end

function apply_transform(rot::SMatrix{2, 2}, pos)
    pos0 = rot * SVector{2}((pos[1], pos[2]))
    pos1 = @set pos[1] = pos0[1]
    pos2 = @set pos1[2] = pos0[2]
    return pos2
end

function apply_transform(transform::AbstractMatrix, pos)
    return transform * pos
end

Base.@propagate_inbounds function get_pos(A::LazyGrid{T, N}, I::Vararg{Int, N}) where {T, N}
    pos0 = SVector{N}(ntuple(n -> rgetindex(A.dirs[n], I[n]), Val(N)))
    pos = apply_transform(A.transform, pos0)
    return Tuple(parent(pos))
end


Base.@propagate_inbounds @inline function Base.getindex(
        A::LazyGrid{T, N, <:NamedTuple{K}},
        I::Vararg{Int, N}
    ) where {T, N, K}
    return NamedTuple{K}(get_pos(A, I...))
end

Base.@propagate_inbounds @inline function Base.getindex(
        A::LazyGrid{T, N, <:Tuple},
        I::Vararg{Int, N}
    ) where {T, N}
    return get_pos(A, I...)
end

@inline getstyle(A1, A2, Arest...) = getstyle(A1, getstyle(A2, Arest...))
@inline getstyle(A1, A2) = Broadcast.BroadcastStyle(getstyle(A1), getstyle(A2))
@inline getstyle(A) = Broadcast.BroadcastStyle(A)
@inline getstyle() = Broadcast.DefaultArrayStyle{0}()

function Base.Broadcast.BroadcastStyle(::Type{<:LazyGrid{T, N, A}}) where {T, N, A}
    inner_style = getstyle(fieldtypes(A)...)
    style = Base.Broadcast.result_style(inner_style, Base.Broadcast.DefaultArrayStyle{N}())
    return style
end
