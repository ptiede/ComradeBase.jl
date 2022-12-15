const StokesIntensityMap{T,N} = KeyedArray{StokesParams{T}, N, <:DataArr} where {T, N}

baseimage(s::KeyedArray) = parent(parent(s))

"""
    StokesIntensityMap(I, Q, U, V)

Create an array of full stokes parameters. The image is stored as a `StructArray` of
@ref[StokesParams].
"""
function StokesIntensityMap(I::IntensityMap{T}, Q::IntensityMap{T}, U::IntensityMap{T}, V::IntensityMap{T}) where {T}
    @assert check_grid(I,Q,U,V) "Intensity grids are not the same across the 4 stokes parameters"
    polimg = StructArray{StokesParams{T}}(I=baseimage(I), Q=baseimage(Q), U=baseimage(U), V=baseimage(V))
    return IntensityMap(polimg, named_axiskeys(I))
end

# # not type piracy because I own StokesParams
# function Base.getproperty(s::StokesIntensityMap{T,N,Na}, v::Symbol) where {T,N,Na<:DataArr}
#     if v ∈ propertynames(s)
#         return axiskeys(s, AxisKeys.NamedDims.dim(Na, v))
#     elseif v ∈ (:I, :Q, :U, :V)
#         return stokes(s, v)
#     else
#         getfield(s, v)
#     end
# end

function StokesIntensityMap(
    I::AbstractArray{T,N}, Q::AbstractArray{T,N},
    U::AbstractArray{T,N}, V::AbstractArray{T,N},
    dims::NamedTuple{Na,<:NTuple{N,Any}}) where {T, N, Na}

    polimg = StructArray{StokesParams{T}}((;I,Q,U,V))
    return IntensityMap(polimg, dims)
end

function StokesIntensityMap(
    I::AbstractArray{T,N}, Q::AbstractArray{T,N},
    U::AbstractArray{T,N}, V::AbstractArray{T,N},
    args...) where {T, N}

    polimg = StructArray{StokesParams{T}}((;I,Q,U,V))
    return IntensityMap(polimg, args...)
end


# simple check to ensure that the four grids are equal across stokes parameters
function check_grid(I,Q,U,V)
    named_axiskeys(I) == named_axiskeys(Q) == named_axiskeys(U) == named_axiskeys(V)
end

function stokes(pimg::StokesIntensityMap, v::Symbol)
    imgb = keyless_uname(pimg)
    imgs = getproperty(imgb, v)
    return IntensityMap(imgs, named_axiskeys(pimg))
end

function Base.summary(io::IO, x::StokesIntensityMap)
    return _summary(io, x)
end

function _summary(io, x::StokesIntensityMap{T,N,Na}) where {T,N,Na}
    println(io, ndims(x), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N, $Na}")
    for d in 1:ndims(x)
        print(io, d==1 ? "↓" : d==2 ? "→" : d==3 ? "◪" : "▨", "   ")
        c = AxisKeys.colour(x, d)
        AxisKeys.hasnames(x) && printstyled(io, AxisKeys.dimnames(x,d), " ∈ ", color=c)
        printstyled(io, length(axiskeys(x,d)), "-element ", AxisKeys.shorttype(axiskeys(x,d)), "\n", color=c)
    end
    println(io, "Polarizations ", propertynames(parent(x).data))
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)
