export UnstructuredMap

"""
    UnstructuredMap(data::AbstractVector, dims::UnstructuredDomain)

A map that is defined on an unstructured domain. This is typically just a vector of values.
The vector of locations of the visibilities are stored in `dims`. Otherwise this behaves
very similarly to `IntensityMap`, except that is isn't a grid.

For instance the locations of the visibilities can be accessed with
`axisdims`, as well as the usual `getproperty` and `propertynames` functions. Like with
`IntensityMap` during execution the `executor` is used to determine the execution context.
"""
struct UnstructuredMap{T, A <: AbstractVector{T}, G <: UnstructuredDomain} <: AbstractVector{T}
    data::A
    dims::G
end
Base.size(x::UnstructuredMap) = (length(x),)
axisdims(x::UnstructuredMap) = getfield(x, :dims)
header(x::UnstructuredMap) = header(axisdims(x))
executor(x::UnstructuredMap) = executor(axisdims(x))
baseimage(x::UnstructuredMap) = baseimage(parent(x))
Base.parent(x::UnstructuredMap) = getfield(x, :data)
Base.length(x::UnstructuredMap) = length(parent(x))
Base.IndexStyle(::Type{<:UnstructuredMap{T, A}}) where {T, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(a::UnstructuredMap, i::Int) = getindex(parent(a), i)
Base.@propagate_inbounds Base.setindex!(a::UnstructuredMap, v, i::Int) = setindex!(
    parent(a),
    v, i
)

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle,
    Style

Base.BroadcastStyle(::Type{<:UnstructuredMap}) = Broadcast.ArrayStyle{UnstructuredMap}()
function Base.similar(
        bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{UnstructuredMap}},
        ::Type{ElType}
    ) where {ElType}
    # Scan inputs for the time and sites
    sarr = find_ustr(bc)
    return UnstructuredMap(similar(parent(sarr), ElType), axisdims(sarr))
end


function Base.copyto!(dest::UnstructuredMap, bc::Broadcast.Broadcasted)
    copyto!(baseimage(dest), bc)
    return dest
end

find_ustr(bc::Broadcasted) = find_ustr(bc.args)
find_ustr(args::Tuple) = find_ustr(find_ustr(args[1]), Base.tail(args))
find_ustr(x) = x
find_ustr(::Tuple{}) = nothing
find_ustr(x::UnstructuredMap, rest) = x
find_ustr(::Any, rest) = find_ustr(rest)

domainpoints(x::UnstructuredMap) = domainpoints(axisdims(x))

Base.propertynames(x::UnstructuredMap) = propertynames(axisdims(x))
Base.getproperty(x::UnstructuredMap, s::Symbol) = getproperty(axisdims(x), s)

# function UnstructuredStyle(a::BroadcastStyle, b::BroadcastStyle)
#     inner_style = BroadcastStyle(a, b)
#     if inner_style isa Broadcast.Unknown
#         return Broadcast.Unknown()
#     else
#         return UnstructuredStyle(inner_style)
#     end
# end

# BroadcastStyle(::UnstructuredStyle, ::Base.Broadcast.Unknown) = Unknown()
# BroadcastStyle(::Base.Broadcast.Unknown, ::UnstructuredStyle) = Unknown()
# BroadcastStyle(::UnstructuredStyle{A}, ::UnstructuredStyle{B}) where {A, B} = UnstructuredStyle(A(), B())
# BroadcastStyle(::UnstructuredStyle{A}, b::Style) where {A} = UnstructuredStyle(A(), b)
# BroadcastStyle(a::Style, ::UnstructuredStyle{B}) where {B} = UnstructuredStyle(a, B())
# BroadcastStyle(::UnstructuredStyle{A}, b::Style{Tuple}) where {A} = UnstructuredStyle(A(), b)
# BroadcastStyle(a::Style{Tuple}, ::UnstructuredStyle{B}) where {B} = UnstructuredStyle(a, B())
# function Base.similar(bc::Broadcasted{UnstructuredStyle},
#     ::Type{ElType}) where {ElType}
# # Scan inputs for the time and sites
# sarr = find_ustr(bc)
# return UnstructuredMap(similar(parent(sarr), ElType), axisdims(sarr))
# end


function UnstructuredMap(data::UnstructuredMap, dims::UnstructuredDomain)
    return UnstructuredMap(parent(data), dims)
end

function stokes(x::UnstructuredMap, p::Symbol)
    return UnstructuredMap(stokes(parent(x), p), axisdims(x))
end

function Base.similar(m::UnstructuredMap, ::Type{S}) where {S}
    return UnstructuredMap(similar(parent(m), S), axisdims(m))
end

function Base.view(x::UnstructuredMap, I)
    dims = axisdims(x)
    g = domainpoints(dims)
    newdims = UnstructuredDomain(@view(g[I]), executor(dims), header(dims))
    return UnstructuredMap(view(parent(x), I), newdims)
end

Base.@propagate_inbounds function Base.getindex(x::UnstructuredMap, I)
    dims = axisdims(x)
    g = domainpoints(dims)
    newdims = UnstructuredDomain((g[I]), executor(dims), header(dims))
    return UnstructuredMap(parent(x)[I], newdims)
end

function intensitymap_analytic!(
        img::UnstructuredMap{T, <:Any, <:AbstractSingleDomain},
        s::AbstractModel
    ) where {T}
    g = domainpoints(img)
    for i in eachindex(img, g)
        img[i] = intensity_point(s, g[i])
    end
    # img .= intensity_point.(Ref(s), g)
    return nothing
end

function intensitymap_analytic!(
        img::UnstructuredMap{
            T, <:Any,
            <:UnstructuredDomain{D, <:ThreadsEx{S}},
        },
        s::AbstractModel
    ) where {T, D, S}
    g = domainpoints(img)
    e = executor(img)
    @threaded e for I in CartesianIndices(g)
        img[I] = intensity_point(s, g[I])
    end
    return nothing
end




# for s in schedulers
#     @eval begin
#         function _threads_intensitymap!(img::UnstructuredMap, s::AbstractModel, g,
#                                         ::Val{$s})
#             Threads.@threads $s for I in CartesianIndices(g)
#                 img[I] = intensity_point(s, g[I])
#             end
#             return nothing
#         end
#     end
# end


