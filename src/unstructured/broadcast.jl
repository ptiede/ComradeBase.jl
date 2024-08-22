using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Style

struct UnstructuredStyle{S<:BroadcastStyle} <: AbstractArrayStyle{1} end
UnstructuredStyle(::S) where {S} = UnstructuredStyle{S}()

Base.BroadcastStyle(::Type{<:UnstructuredMap{T,A}}) where {T,A} = UnstructuredStyle(A)
function UnstructuredStyle(a::BroadcastStyle, b::BroadcastStyle)
    inner_style = BroadcastStyle(a, b)
    if inner_style isa Broadcast.Unknown
        return Broadcast.Unknown()
    else
        return UnstructuredStyle(inner_style)
    end
end

Broadcast.BroadcastStyle(::UnstructuredStyle{A}, ::UnstructuredStyle{B}) where {A, B} = UnstructuredStyle(A(), B())
Broadcast.BroadcastStyle(::UnstructuredStyle{A}, b::Style) where {A} = UnstructuredStyle(A(), b)
Broadcast.BroadcastStyle(a::Style, ::UnstructuredStyle{B}) where {B} = UnstructuredStyle(a, B())
Broadcast.BroadcastStyle(::UnstructuredStyle{A}, b::Style{Tuple}) where {A} = UnstructuredStyle(A(), b)
Broadcast.BroadcastStyle(a::Style{Tuple}, ::UnstructuredStyle{B}) where {B} = UnstructuredStyle(a, B())


function Base.similar(bc::Broadcasted{UnstructuredStyle},
                      ::Type{ElType}) where {ElType}
    # Scan inputs for the time and sites
    sarr = find_ustr(bc)
    return UnstructuredMap(similar(parent(sarr), ElType), axisdims(sarr))
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
