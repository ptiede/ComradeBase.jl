export UnstructuredMap

struct UnstructuredMap{T, A<:AbstractVector{T}, G<:UnstructuredGrid} <: AbstractVector{T}
    data::A
    dims::G
end
Base.size(x::UnstructuredMap) = (length(x),)
axisdims(x::UnstructuredMap) = getfield(x, :dims)
header(x::UnstructuredMap) = header(axisdims(x))
executor(x::UnstructuredMap) = executor(axisdims(x))
baseimage(x::UnstructuredMap) = parent(x)
Base.parent(x::UnstructuredMap) = getfield(x, :data)
Base.length(x::UnstructuredMap) = length(parent(x))
Base.IndexStyle(::Type{<:UnstructuredMap{T, A}}) where {T, A} = IndexStyle(A)
Base.getindex(a::UnstructuredMap, i::Int) = getindex(parent(a), i)
Base.setindex!(a::UnstructuredMap, v, i::Int) = setindex!(parent(a), v, i)

UnstructuredMap(data::UnstructuredMap, dims::AbstractGrid) = UnstructuredMap(parent(data), dims)

function Base.similar(m::UnstructuredMap, ::Type{S}) where {S}
    return UnstructuredMap(similar(parent(m), S), axisdims(m))
end

Base.BroadcastStyle(::Type{<:UnstructuredMap}) = Broadcast.ArrayStyle{UnstructuredMap}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{UnstructuredMap}}, ::Type{ElType}) where {ElType}
    # Scan inputs for the time and sites
    sarr = find_ustr(bc)
    return UnstructuredMap(similar(parent(sarr), ElType), axisdims(sarr))
end

find_ustr(bc::Broadcast.Broadcasted) = find_ustr(bc.args)
find_ustr(args::Tuple) = find_ustr(find_ustr(args[1]), Base.tail(args))
find_ustr(x) = x
find_ustr(::Tuple{}) = nothing
find_ustr(x::UnstructuredMap, rest) = x
find_ustr(::Any, rest) = find_ustr(rest)


function Base.view(x::UnstructuredMap, I)
    dims = axisdims(x)
    g = imagegrid(dims)
    newdims = ucturedGrid(@view(g[I]), executor(dims), header(dims))
    UnstructuredMap(view(parent(x), I), newdims)
end

function Base.getindex(x::UnstructuredMap, I)
    dims = axisdims(x)
    g = imagegrid(dims)
    newdims = UnstructuredGrid((g[I]), executor(dims), header(dims))
    UnstructuredMap(parent(x)[I], newdims)
end

function intensitymap_analytic(m::AbstractModel, dims::UnstructuredGrid)
    g = imagegrid(dims)
    img = intensity_point.(Ref(m), g)
    return UnstructuredMap(img, dims)
end

function intensitymap_analytic!(img::UnstructuredMap, s)
    g = imagegrid(img)
    img .= intensity_point.(Ref(s), g)
end


function intensitymap_analytic(s::AbstractModel, dims::UnstructuredGrid{D, <:ThreadsEx}) where {D}
    img = UnstructuredMap(zeros(eltype(dims), size(dims)), dims)
    intensitymap_analytic!(img, s)
    return img
end

function intensitymap_analytic!(
    img::UnstructuredMap{T,<:AbstractVector,<:UnstructuredGrid{D, <:ThreadsEx{S}}},
    s::AbstractModel) where {T,D,S}
    g = imagegrid(img)
    _threads_intensitymap!(img, s, g, Val(S))
    return img
end

for s in schedulers
    @eval begin
        function _threads_intensitymap!(img::UnstructuredMap, s::AbstractModel, g, ::Val{$s})
            Threads.@threads $s for I in CartesianIndices(g)
                img[I] = intensity_point(s, g[I])
            end
        end
    end
end
