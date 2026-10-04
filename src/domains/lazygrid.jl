struct LazyGrid{T, N, Dirs <: NamedTuple, TR <: SMatrix{2, 2}} <: AbstractArray{T, N}
    dirs::Dirs
    transform::TR
    @inline function LazyGrid(dirs::NamedTuple, transform)
        T = geteltype(typeof(dirs))
        N = length(dirs)
        return new{T, N, typeof(dirs), typeof(transform)}(dirs, transform)
    end
end

@inline function geteltype(::Type{<:NamedTuple{N, A}}) where {N, A}
    return NamedTuple{N, geteltype(A)}
end

@inline function geteltype(T::Type{<:Tuple})
    et = map(eltype, fieldtypes(T))
    return Tuple{et...}
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

function apply_transform(rot::SMatrix{2, 2}, pos)
    pos0 = rot * SVector{2}((pos[1], pos[2]))
    pos1 = @set pos[1] = pos0[1]
    pos2 = @set pos1[2] = pos0[2]
    return pos2
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

@inline getstyle() = Broadcast.DefaultArrayStyle{0}()
@inline function getstyle(A, Arest...)
    return Broadcast.result_style(Broadcast.BroadcastStyle(A), getstyle(Arest...))
end

function Base.Broadcast.BroadcastStyle(::Type{<:LazyGrid{T, N, A}}) where {T, N, A}
    inner_style = getstyle(fieldtypes(A)...)
    style = Base.Broadcast.result_style(inner_style, Base.Broadcast.DefaultArrayStyle{N}())
    return style
end
