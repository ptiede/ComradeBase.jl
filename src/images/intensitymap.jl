export IntensityMap, fov, imagepixels, pixelsizes, stokes, centroid, inertia, phasecenter

"""
    ContinuousImage{A<:AbstractDimArray, P}


"""
struct ContinuousImage{A <: KeyedArray, P} <: AbstractModel
    """
    Discrete representation of the image. This must be a DimArray where at least two of the
    """
    img::S
    """
    field of view first is in x direction second in y
    """
    fov::F
    """
    pixel sizes in the x, y direction
    """
    psize::F
    """
    phase center offset of the image
    """
    phasecenter::F
    """
    pulse function that turns the image grid into a continuous object
    """
    pulse::K
end

function IntensityMap(im::AbstractMatrix, fov::NTuple{2}, phasecenter=(0.0, 0.0), pulse=DeltaPulse())
    ny,nx = size(im)
    fovx, fovy = fov
    psizex=fovx/max(nx-1,1)
    psizey=fovy/max(ny-1,1)
    psize = (psizex, psizey)
    F = promote_type(eltype(fov), eltype(phasecenter))
    TF = Tuple{F,F}
    return IntensityMap{eltype(im), typeof(im), TF, typeof(pulse)}(im,
                       convert.(F, fov),
                       psize,
                       convert.(F,phasecenter),
                       pulse
                       )
end

function IntensityMap(im::AbstractMatrix, fovx::Real, fovy::Real, phasecenter=(0.0, 0.0), pulse=DeltaPulse())
    return IntensityMap(im, (fovx, fovy), phasecenter, pulse)
end

function IntensityMap(im::AbstractMatrix, fovx::Real, phasecenter=(0.0, 0.0), pulse=DeltaPulse())
    return IntensityMap(im, (fovx, fovx), phaecenter, pulse)
end

Base.getindex(img::ContinuousImage, args...) = getindex(img.img, args...)
Base.setindex!(img::ContinuousImage, args...) = setindex!(img.img, args...)


imagepixels(img::ContinuousImage) = NamedTuple{names.(dims(img.img))}(dims(img.img))

# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()
isprimitive(::Type{<:ContinuousImage}) = IsPrimitive()

function intensity_point(m::ContinuousImage, p)
    dx, dy = pixelsizes(m)
    sum = zero(eltype(m.img))
    @inbounds for (I, p0) in pairs(grid(m.img))
        dp = (p.X - p0.X, p.Y - p0.Y)
        k = intensity_point(m.pulse, dp)
        sum += m.img[I]*k/(dx*dy)
    end
    return sum
end


function IntensityMap(img::Matrix{<:StokesVector}, fov, phasecenter, pulse)
    return IntensityMap(StructArray(img), fov, phasecenter, pulse)
end

function IntensityMap(img::IntensityMap{<:StokesVector}, fov, phasecenter, pulse)
    return IntensityMap(img, fov, phasecenter, pulse)
end

function stokes(img::IntensityMap{<:StokesVector}, p::Symbol)
    @assert p ∈ propertynames(img.img) "$p is not a valid stokes parameter"
    return IntensityMap(getproperty(img.img, p), fov(img), img.phasecenter, img.pulse)
end

function IntensityMap(I::AbstractIntensityMap,
                      Q::AbstractIntensityMap,
                      U::AbstractIntensityMap,
                      V::AbstractIntensityMap
                      )
    @assert I.fov == Q.fov == U.fov == V.fov "Must have matching field of view"
    @assert I.pulse == Q.pulse == U.pulse == V.pulse "Must have matching pulses"
    @assert I.phasecenter == Q.phasecenter == U.phasecenter == V.phasecenter "Must have matching phasecenter"
    simg = StructArray{StokesVector{eltype(I)}}((I,Q,U,V))
    return IntensityMap(simg, fov(I), I.phasecenter, I.pulse)
end


# const SpatialOnly = Union{Tuple{<:X, <:Y}, Tuple{<:Y, <:X}}

# function ChainRulesCore.rrule(::Type{<:IntensityMap}, im, fovx, fovy, pulse)
#     y = IntensityMap(im, fovx, fovy, pulse)
#     intensity_pullback(Δy) = (NoTangent(), IntensityMap(Δy.im, fovx, fovy, pulse), Δy.fovx, Δy.fovy, Δy.pulse)
#     return y, intensity_pullback
# end

"""
    fov(img::AbstractIntensityMap)

Returns the field of view (fov) of the image `img` as a Tuple
where the first element is in the RA direction and the second the DEC.
"""
fov(m::AbstractIntensityMap) = m.fov

"""
    psizes(img::AbstractIntensityMap)

Returns the pixel sizes of the image `img` as a Tuple
where the first element is in the RA direction and the second the DEC.
"""
psizes(img::AbstractIntensityMap) = img.psize

"""
    phasecenter(img::AbstractIntensityMap)

Returns the phase center of the image `img` as a Tuple
where the first element is in the RA direction and the second the DEC.
"""
phasecenter(img::AbstractIntensityMap) = img.phasecenter

"""
    flux(im::AbstractIntensityMap)

Computes the flux of a intensity map
"""
function flux(im::AbstractIntensityMap{T,S}) where {T,S}
    f = sum(im.img)#*(flux(im.pulse))^2
    return f#*prod(pixelsizes(im))
end

"""
    phasecenter(img::AbstractIntensityMap)

Returns the phase center of the image `img` as a Tuple
where the first element is in the RA direction and the second the DEC.
"""
phasecenter(img::AbstractIntensityMap) = img.phasecenter


# Define the array interface
Base.IndexStyle(::Type{<: AbstractIntensityMap{T,S}}) where {T,S} = Base.IndexStyle(S)
Base.size(im::AbstractIntensityMap) = size(im.img)
Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, i::Int) = getindex(im.img, i)
Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, I...) = getindex(im.img, I...)
Base.@propagate_inbounds Base.setindex!(im::AbstractIntensityMap, x, i::Int) = setindex!(im.img, x, i)
Base.@propagate_inbounds Base.setindex!(im::AbstractIntensityMap, x, i) = setindex!(im.img, x, i)

function Base.similar(im::IntensityMap, ::Type{T}) where{T}
    sim = similar(im.img, T)
    return IntensityMap(sim, fov(im), im.phasecenter, im.pulse)
end

# #function Base.similar(im::AbstractIntensityMap, ::Type{T}, dims::Dims) where {T}
# #    fovx = im.psizex*last(dims)
# #    fovy = im.psizey*first(dims)
# #    sim = similar(im.im, T, dims)
# #    return IntensityMap(sim, fovx, fovy, im.psizex, im.psizey)
# #end

# #Define the broadcast interface
# struct IntensityMapStyle <: Broadcast.AbstractArrayStyle{2} end
# IntensityMapStyle(::Val{2}) = IntensityMapStyle()

Base.BroadcastStyle(::Type{<:AbstractIntensityMap}) = IntensityMapStyle()
function Base.similar(bc::Broadcast.Broadcasted{IntensityMapStyle}, ::Type{ElType}) where ElType
    #Scan inputs for IntensityMap
    #print(bc.args)
    Im = _find_sim(bc)
    #fovxs = getproperty.(Im, Ref(:fovx))
    #fovys = getproperty.(Im, Ref(:fovy))
    #@assert all(i->i==first(fovxs), fovxs) "IntensityMap fov must be equal to add"
    #@assert all(i->i==first(fovys), fovys) "IntensityMap fov must be equal to add"
    return IntensityMap(similar(Array{ElType}, axes(bc)), fov(Im), Im.phasecenter, Im.pulse)
end

#Finds the first IntensityMap and uses that as the base
#TODO: If multiply IntensityMaps maybe I should make sure they are consistent?
_find_sim(bc::Base.Broadcast.Broadcasted) = _find_sim(bc.args)
_find_sim(args::Tuple) = _find_sim(_find_sim(args[1]), Base.tail(args))
_find_sim(x) = x
_find_sim(::Tuple{}) = nothing
_find_sim(a::AbstractIntensityMap, _) = a
_find_sim(::Any, rest) = _find_sim(rest)

#Guards to prevent someone from adding two Images with different FOV's
function Base.:+(x::AbstractIntensityMap, y::AbstractIntensityMap)
    @assert fov(x) == fov(y) "IntensityMaps must share same field of view"
    @assert phasecenter(x) == phasecenter(y) "IntensityMaps must share same phasecenter"
    return x .+ y
end


@inline function pixelsizes(im::AbstractIntensityMap)
    ny,nx = size(im)
    fovx, fovy = fov(im)
    return fovx/nx, fovy/ny
end

@inline function imagepixels(fovx, fovy, x0, y0, nx::Int, ny::Int)
    px = fovx/nx; py = fovy/ny
    return range(-fovx/2+px/2 - x0, step=px, length=nx),
           range(-fovy/2+py/2 - y0, step=py, length=ny)
end

@inline function imagepixels(im::AbstractIntensityMap)
    ny,nx = size(im)
    x0 ,y0 = phasecenter(im)
    fovx, fovy = fov(im)
    return imagepixels(fovx, fovy, x0, y0, nx, ny)
end
