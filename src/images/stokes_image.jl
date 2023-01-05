export StokesIntensityMap, keyless_unname

struct StokesIntensityMap{T, N, S<:IntensityMap{T, N}}
    """
    Stokes I image
    """
    I::S
    """
    Stokes Q image
    """
    Q::S
    """
    Stokes U image
    """
    U::S
    """
    Stokes V image
    """
    V::S
    function StokesIntensityMap(
        I::S,
        Q::S,
        U::S,
        V::S) where {T<:Real, N, S<:IntensityMap{T,N}}

        check_grid(I, Q, U, V)
        return new{T, N, S}(I, Q, U, V)
    end
end

Base.size(im::StokesIntensityMap) = size(im.I)
Base.eltype(::StokesIntensityMap{T}) where {T} = StokesParams{T}
Base.ndims(::StokesIntensityMap{T,N}) where {T,N} = N
Base.@propagate_inbounds Base.getindex(im::StokesIntensityMap, i::Int) = StokesParams(getindex(im.I, i),getindex(im.Q, i),getindex(im.U, i),getindex(im.V, i))
Base.@propagate_inbounds Base.getindex(im::StokesIntensityMap, I...) = StokesParams.(getindex(im.I, I...), getindex(im.Q, I...), getindex(im.U, I...), getindex(im.V, I...))

function Base.setindex!(im::StokesIntensityMap, x::StokesParams, inds...)
    setindex!(im.I, x.I, inds...)
    setindex!(im.Q, x.Q, inds...)
    setindex!(im.U, x.U, inds...)
    setindex!(im.V, x.V, inds...)
end

pixelsizes(img::StokesIntensityMap)  = pixelsizes(img.I)
imagepixels(img::StokesIntensityMap) = imagepixels(img.I)
fieldofview(img::StokesIntensityMap) = fieldofview(img.I)
imagegrid(img::StokesIntensityMap)   = imagegrid(img.I)

function StokesIntensityMap(
    I::AbstractArray{T,N}, Q::AbstractArray{T,N},
    U::AbstractArray{T,N}, V::AbstractArray{T,N},
    dims::NamedTuple{Na,<:NTuple{N,Any}}) where {T, N, Na}

    imgI = IntensityMap(I, dims)
    imgQ = IntensityMap(Q, dims)
    imgU = IntensityMap(U, dims)
    imgV = IntensityMap(V, dims)
    return StokesIntensityMap(imgI, imgQ, imgU, imgV)
end


# simple check to ensure that the four grids are equal across stokes parameters
function check_grid(I::IntensityMap, Q::IntensityMap,U::IntensityMap ,V::IntensityMap)
    named_axiskeys(I) == named_axiskeys(Q) == named_axiskeys(U) == named_axiskeys(V)
end

function stokes(pimg::StokesIntensityMap, v::Symbol)
    return getproperty(pimg, v)
end



function Base.summary(io::IO, x::StokesIntensityMap)
    return _summary(io, x)
end

function _summary(io, x::StokesIntensityMap{T,N}) where {T,N}
    println(io, ndims(x), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N}")
    println(io, "   Stokes I: ")
    summary(io, x.I)
    println(io, "   Stokes Q: ")
    summary(io, x.Q)
    println(io, "   Stokes U: ")
    summary(io, x.U)
    println(io, "   Stokes V: ")
    summary(io, x.V)
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)
