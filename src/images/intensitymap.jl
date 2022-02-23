export IntensityMap, fov, imagepixels, pixelsizes, stokes

"""
    $(TYPEDEF)
Image array type. This is an Matrix with a number of internal fields
to describe the field of view, pixel size, and the `pulse` function that
makes the image a continuous quantity.

To use it you just specify the array and the field of view/pulse
``julia
img = IntensityMap(zeros(512,512), 100.0, 100.0)
```
"""
struct IntensityMap{T,S<:AbstractMatrix, F, K<:Pulse} <: AbstractIntensityMap{T,S}
    im::S
    fovx::F
    fovy::F
    psizex::F
    psizey::F
    pulse::K
end

function IntensityMap(im, fovx, fovy, pulse=DeltaPulse())
    ny,nx = size(im)
    psizex=fovx/(nx-1)
    psizey=fovy/(ny-1)
    F = promote_type(typeof(fovx), typeof(fovy))
    return IntensityMap{eltype(im), typeof(im), F, typeof(pulse)}(im,
                       convert(typeof(psizex),fovx),
                       convert(typeof(psizey),fovy),
                       psizex,
                       psizey,
                       pulse)
end


function IntensityMap(im::Matrix{<:StokesVector}, fovx, fovy, pulse)
    return IntensityMap(StructArray(im), fovx, fovy, pulse)
end

function IntensityMap(im::IntensityMap{<:StokesVector}, fovx, fovy, pulse)
    return IntensityMap(im.im, fovx, fovy, pulse)
end

function stokes(im::IntensityMap{<:StokesVector}, p::Symbol)
    @assert p ∈ propertynames(im.im) "$p is not a valid stokes parameter"
    return IntensityMap(getproperty(im.im, p), im.fovx, im.fovy, im.pulse)
end

function IntensityMap(I::AbstractIntensityMap,
                      Q::AbstractIntensityMap,
                      U::AbstractIntensityMap,
                      V::AbstractIntensityMap
                      )
    @assert I.fovx == Q.fovx == U.fovx == V.fovx "Must have matching fov in RA"
    @assert I.fovy == Q.fovy == U.fovy == V.fovy "Must have matching fov in DEC"
    simg = StructArray{StokesVector{eltype(I)}}((I,Q,U,V))
    return IntensityMap(simg, I.fovx, I.fovy, I.pulse)
end



function ChainRulesCore.rrule(::Type{<:IntensityMap}, im, fovx, fovy, pulse)
    y = IntensityMap(im, fovx, fovy, pulse)
    intensity_pullback(Δy) = (NoTangent(), Δy, fovx, fovy, pulse)
    return y, intensity_pullback
end

fov(m::AbstractIntensityMap) = (m.fovx, m.fovy)




"""
    $(SIGNATURES)
Computes the flux of a intensity map
"""
function flux(im::AbstractIntensityMap{T,S}) where {T,S}
    sum = zero(T)
    x,y = imagepixels(im)
    @inbounds for i in axes(im,1), j in axes(im, 2)
        xx = x[i]
        yy = y[j]
        sum += im[j,i]*ComradeBase.intensity_point(im.pulse, xx, yy)
    end
    return sum*prod(pixelsizes(im))
end

"""
    $(SIGNATURES)
Computes the flux of a intensity map
"""
function flux(im::AbstractIntensityMap{T,S}) where {F,T<:StokesVector{F},S}
    sum = zero(F)
    x,y = imagepixels(im)
    I = stokes(im, :I)
    @inbounds for i in axes(im,1), j in axes(im, 2)
        xx = x[i]
        yy = y[j]
        sum += I[j,i]*ComradeBase.intensity_point(im.pulse, xx, yy)
    end
    return sum*prod(pixelsizes(im))
end



# Define the array interface
Base.IndexStyle(::Type{<: IntensityMap{T,S,K}}) where {T,S,K} = Base.IndexStyle(S)
Base.size(im::AbstractIntensityMap) = size(im.im)
Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, i::Int) = getindex(im.im, i)
Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, I...) = getindex(im.im, I...)
Base.@propagate_inbounds Base.setindex!(im::AbstractIntensityMap, x, i::Int) = setindex!(im.im, x, i)

function Base.similar(im::IntensityMap, ::Type{T}) where{T}
    sim = similar(im.im, T)
    return IntensityMap(sim, im.fovx, im.fovy, im.pulse)
end

#function Base.similar(im::AbstractIntensityMap, ::Type{T}, dims::Dims) where {T}
#    fovx = im.psizex*last(dims)
#    fovy = im.psizey*first(dims)
#    sim = similar(im.im, T, dims)
#    return IntensityMap(sim, fovx, fovy, im.psizex, im.psizey)
#end

#Define the broadcast interface
struct IntensityMapStyle <: Broadcast.AbstractArrayStyle{2} end
IntensityMapStyle(::Val{2}) = IntensityMapStyle()

Base.BroadcastStyle(::Type{<:AbstractIntensityMap}) = IntensityMapStyle()
function Base.similar(bc::Broadcast.Broadcasted{IntensityMapStyle}, ::Type{ElType}) where ElType
    #Scan inputs for IntensityMap
    #print(bc.args)
    Im = _find_sim(bc)
    #fovxs = getproperty.(Im, Ref(:fovx))
    #fovys = getproperty.(Im, Ref(:fovy))
    #@assert all(i->i==first(fovxs), fovxs) "IntensityMap fov must be equal to add"
    #@assert all(i->i==first(fovys), fovys) "IntensityMap fov must be equal to add"
    return IntensityMap(similar(Array{ElType}, axes(bc)), Im.fovx, Im.fovy, Im.pulse)
end

#Finds the first IntensityMap and uses that as the base
#TODO: If multiply IntensityMaps maybe I should make sure they are consistent?
_find_sim(bc::Base.Broadcast.Broadcasted) = _find_sim(bc.args)
_find_sim(args::Tuple) = _find_sim(_find_sim(args[1]), Base.tail(args))
_find_sim(x) = x
_find_sim(::Tuple{}) = nothing
_find_sim(a::AbstractIntensityMap, rest) = a
_find_sim(::Any, rest) = _find_sim(rest)

#Guards to prevent someone from adding two Images with different FOV's
function Base.:+(x::AbstractIntensityMap, y::AbstractIntensityMap)
    @assert fov(x) == fov(y) "IntensityMaps must share same field of view"
    return x .+ y
end


@inline function pixelsizes(im::AbstractIntensityMap)
    ny,nx = size(im)
    return im.fovx/nx, im.fovy/ny
end

@inline function imagepixels(fovx, fovy, nx::Int, ny::Int)
    px = fovx/nx; py = fovy/ny
    return range(-fovx/2+px/2, step=px, length=nx),
           range(-fovy/2+py/2, step=py, length=ny)
end

@inline function imagepixels(im::AbstractIntensityMap)
    ny,nx = size(im)
    return imagepixels(im.fovx, im.fovy, nx, ny)
end
