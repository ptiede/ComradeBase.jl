using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle,
                      Style

Base.BroadcastStyle(::Type{<:UnstructuredMap}) = Broadcast.ArrayStyle{UnstructuredMap}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{UnstructuredMap}},
                      ::Type{ElType}) where {ElType}
    # Scan inputs for the time and sites
    sarr = find_ustr(bc)
    return UnstructuredMap(similar(parent(sarr), ElType), axisdims(sarr))
end

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
